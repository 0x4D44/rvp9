//! Real-time encoding latency benchmark.
//!
//! Drives a configurable encoder binary (defaults to libvpx vpxenc) against a
//! YUV4MPEG (.y4m) input file at a target speed setting and reports aggregate
//! per-frame encode wall-time: total time, frame count, average ms/frame.
//! When `--frame-time-csv` is supplied to an encoder that supports the
//! rvp9enc benchmark diagnostic, it also reports encoder-only p50/p90/p99
//! frame times from the exact CSV rows.
//!
//! In Stage 0 this runs `libvpx vpxenc` as our baseline and validates that
//! the harness works. In Stage 4+, the same binary will be re-pointed at
//! `rvp9enc` to gate real-time compliance.
//!
//! Usage:
//!   cargo run --release --bin realtime -- \
//!       --encoder target/release/rvp9enc \
//!       --input   tests/corpus/parkrun_720p30.y4m \
//!       --cpu-used 7 \
//!       --target-bitrate 2000 \
//!       --budget-ms 33 \
//!       --frame-time-csv /tmp/parkrun-frame-times.csv \
//!       --p99-budget-ms 50
//!
//! The encoder matrix controls (`--end-usage`, `--threads`,
//! `--tile-columns`, `--temporal-layers`, `--spatial-layers`,
//! `--inter-layer-pred`, `--aq-mode`, `--noise-sensitivity`, `--limit`, and
//! `--kf-max-dist`) are forwarded in vpxenc-compatible form. Defaults are CBR,
//! one thread, zero tile columns, AQ mode 0, denoiser 0, and keyframe distance
//! 60; temporal and spatial layers are omitted unless explicitly selected.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use rvp9_bench::{Stats, count_ivf_frames};

const FRAME_TIME_CSV_HEADER: &str = "frame_index,temporal_layer_id,encode_ns";

struct Args {
    encoder: PathBuf,
    input: PathBuf,
    cpu_used: u32,
    target_bitrate_kbps: u32,
    budget_ms: f64,
    output: PathBuf,
    end_usage: String,
    threads: u32,
    tile_columns: u8,
    temporal_layers: Option<u32>,
    spatial_layers: Option<u32>,
    inter_layer_pred: u32,
    aq_mode: u32,
    noise_sensitivity: u32,
    limit: Option<usize>,
    kf_max_dist: u32,
    frame_time_csv: Option<PathBuf>,
    p99_budget_ms: Option<f64>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    parse_args_from(&argv)
}

fn usage() -> String {
    "Usage: realtime --input CLIP.y4m [options]\n\n\
Options:\n  --encoder PATH --output PATH\n  \
--cpu-used {6,7,8} --target-bitrate KBPS --end-usage {cbr,vbr,cq,q}\n  \
--threads 1..256 --tile-columns 0..6\n  \
--temporal-layers {2,3} | --spatial-layers {2,3} [--inter-layer-pred {0,3}]\n  \
--aq-mode {0,3} --noise-sensitivity 0..6\n  \
--limit N --kf-max-dist N\n  \
--budget-ms 33.333\n  \
--frame-time-csv PATH [--p99-budget-ms MS]\n"
        .to_owned()
}

/// A capability probe: `--help` / `-h` anywhere in argv answers with usage and
/// exits 0, so a caller can discover the flag surface without supplying
/// `--input` or any other required argument.
fn help_requested(argv: &[String]) -> bool {
    argv.iter().any(|arg| arg == "--help" || arg == "-h")
}

fn parse_args_from(argv: &[String]) -> Args {
    let get = |flag: &str| -> Option<String> {
        argv.iter().enumerate().find_map(|(i, arg)| {
            if arg == flag {
                argv.get(i + 1).cloned()
            } else {
                arg.strip_prefix(flag)
                    .and_then(|value| value.strip_prefix('='))
                    .map(str::to_owned)
            }
        })
    };

    let encoder = get("--encoder")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("vpxenc.exe"));
    let input = get("--input")
        .map(PathBuf::from)
        .expect("--input <path.y4m> is required");
    let cpu_used = get("--cpu-used")
        .map(|s| s.parse::<u32>().expect("--cpu-used must be u32"))
        .unwrap_or(7);
    assert!(
        (6..=8).contains(&cpu_used),
        "--cpu-used must be in the focused realtime range 6..=8"
    );
    let target_bitrate_kbps = get("--target-bitrate")
        .map(|s| s.parse::<u32>().expect("--target-bitrate must be u32"))
        .unwrap_or(2000);
    let budget_ms = get("--budget-ms")
        .map(|s| s.parse::<f64>().expect("--budget-ms must be f64"))
        .unwrap_or(33.333);
    let output = get("--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rvp9_realtime_bench.ivf"));
    let end_usage = get("--end-usage").unwrap_or_else(|| "cbr".to_string());
    assert!(
        matches!(end_usage.as_str(), "cbr" | "vbr" | "cq" | "q"),
        "--end-usage must be one of cbr, vbr, cq, q"
    );
    let threads = get("--threads")
        .map(|s| s.parse::<u32>().expect("--threads must be u32"))
        .unwrap_or(1);
    assert!(
        (1..=256).contains(&threads),
        "--threads must be in the range 1..=256"
    );
    let tile_columns = get("--tile-columns")
        .map(|s| s.parse::<u8>().expect("--tile-columns must be u8"))
        .unwrap_or(0);
    assert!(
        tile_columns <= 6,
        "--tile-columns must be in the range 0..=6"
    );
    let temporal_layers = get("--temporal-layers").map(|s| {
        let layers = s.parse::<u32>().expect("--temporal-layers must be u32");
        assert!(
            (2..=3).contains(&layers),
            "--temporal-layers must be 2 or 3 when provided"
        );
        layers
    });
    let spatial_layers = get("--spatial-layers").map(|s| {
        let layers = s.parse::<u32>().expect("--spatial-layers must be u32");
        assert!(
            matches!(layers, 2 | 3),
            "--spatial-layers must be 2 or 3 when provided"
        );
        layers
    });
    assert!(
        temporal_layers.is_none() || spatial_layers.is_none(),
        "--temporal-layers and --spatial-layers cannot be combined"
    );
    let inter_layer_pred = get("--inter-layer-pred")
        .map(|s| s.parse::<u32>().expect("--inter-layer-pred must be u32"))
        .unwrap_or(0);
    assert!(
        matches!(inter_layer_pred, 0 | 3),
        "--inter-layer-pred must be 0 or 3"
    );
    let aq_mode = get("--aq-mode")
        .map(|s| s.parse::<u32>().expect("--aq-mode must be u32"))
        .unwrap_or(0);
    assert!(matches!(aq_mode, 0 | 3), "--aq-mode must be 0 or 3");
    let noise_sensitivity = get("--noise-sensitivity")
        .map(|s| s.parse::<u32>().expect("--noise-sensitivity must be u32"))
        .unwrap_or(0);
    assert!(
        noise_sensitivity <= 6,
        "--noise-sensitivity must be in the range 0..=6"
    );
    let limit = get("--limit").map(|s| s.parse::<usize>().expect("--limit must be usize"));
    let kf_max_dist = get("--kf-max-dist")
        .map(|s| s.parse::<u32>().expect("--kf-max-dist must be u32"))
        .unwrap_or(60);
    assert!(kf_max_dist > 0, "--kf-max-dist must be greater than zero");
    let frame_time_csv = get("--frame-time-csv").map(PathBuf::from);
    let p99_budget_ms = get("--p99-budget-ms").map(|s| {
        let budget = s.parse::<f64>().expect("--p99-budget-ms must be f64");
        assert!(
            budget.is_finite() && budget > 0.0,
            "--p99-budget-ms must be finite and greater than zero"
        );
        budget
    });
    assert!(
        p99_budget_ms.is_none() || frame_time_csv.is_some(),
        "--p99-budget-ms requires --frame-time-csv"
    );

    Args {
        encoder,
        input,
        cpu_used,
        target_bitrate_kbps,
        budget_ms,
        output,
        end_usage,
        threads,
        tile_columns,
        temporal_layers,
        spatial_layers,
        inter_layer_pred,
        aq_mode,
        noise_sensitivity,
        limit,
        kf_max_dist,
        frame_time_csv,
        p99_budget_ms,
    }
}

fn parse_frame_time_csv(text: &str, expected_frames: usize) -> Result<Stats, String> {
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Err("timing CSV is empty".to_string());
    };
    if header != FRAME_TIME_CSV_HEADER {
        return Err(format!(
            "timing CSV header must be `{FRAME_TIME_CSV_HEADER}`, got `{header}`"
        ));
    }

    let mut samples_ms = Vec::new();
    for (row_offset, line) in lines.enumerate() {
        let line_number = row_offset + 2;
        if line.is_empty() {
            return Err(format!("timing CSV row {line_number} is empty"));
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 3 {
            return Err(format!(
                "timing CSV row {line_number} must have exactly 3 fields"
            ));
        }
        let frame_index = fields[0].parse::<usize>().map_err(|_| {
            format!(
                "timing CSV row {line_number} has invalid frame_index `{}`",
                fields[0]
            )
        })?;
        if frame_index != samples_ms.len() {
            return Err(format!(
                "timing CSV row {line_number} has frame_index {frame_index}, expected {}",
                samples_ms.len()
            ));
        }
        let temporal_layer_id = fields[1].parse::<u8>().map_err(|_| {
            format!(
                "timing CSV row {line_number} has invalid temporal_layer_id `{}`",
                fields[1]
            )
        })?;
        if temporal_layer_id > 2 {
            return Err(format!(
                "timing CSV row {line_number} has temporal_layer_id {temporal_layer_id}, expected 0..=2"
            ));
        }
        let encode_ns = fields[2].parse::<u128>().map_err(|_| {
            format!(
                "timing CSV row {line_number} has invalid encode_ns `{}`",
                fields[2]
            )
        })?;
        if encode_ns == 0 {
            return Err(format!("timing CSV row {line_number} has encode_ns=0"));
        }
        let encode_ms = encode_ns as f64 / 1_000_000.0;
        if !encode_ms.is_finite() {
            return Err(format!(
                "timing CSV row {line_number} encode_ns is too large"
            ));
        }
        samples_ms.push(encode_ms);
    }

    if samples_ms.is_empty() {
        return Err("timing CSV has a header but no data rows".to_string());
    }
    if samples_ms.len() != expected_frames {
        return Err(format!(
            "timing CSV has {} data rows, but IVF contains {expected_frames} frames",
            samples_ms.len()
        ));
    }
    samples_ms.sort_by(f64::total_cmp);
    Stats::from_sorted(&samples_ms).ok_or_else(|| "timing CSV has no samples".to_string())
}

fn read_frame_time_csv(path: &std::path::Path, expected_frames: usize) -> Result<Stats, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("reading timing CSV {}: {e}", path.display()))?;
    parse_frame_time_csv(&text, expected_frames)
}

fn encoder_args(args: &Args) -> Vec<OsString> {
    let mut command_args: Vec<OsString> = vec![
        "--codec=vp9".into(),
        "--passes=1".into(),
        "--rt".into(),
        format!("--cpu-used={}", args.cpu_used).into(),
        format!("--end-usage={}", args.end_usage).into(),
        format!("--target-bitrate={}", args.target_bitrate_kbps).into(),
        format!("--threads={}", args.threads).into(),
        format!("--tile-columns={}", args.tile_columns).into(),
        "--lag-in-frames=0".into(),
        "--tune-content=default".into(),
        format!("--aq-mode={}", args.aq_mode).into(),
        format!("--noise-sensitivity={}", args.noise_sensitivity).into(),
        format!("--kf-max-dist={}", args.kf_max_dist).into(),
    ];
    if let Some(layers) = args.temporal_layers {
        command_args.push(format!("--temporal-layers={layers}").into());
    }
    if let Some(layers) = args.spatial_layers {
        command_args.push(format!("--spatial-layers={layers}").into());
        command_args.push(format!("--inter-layer-pred={}", args.inter_layer_pred).into());
    }
    if let Some(limit) = args.limit {
        command_args.push(format!("--limit={limit}").into());
    }
    command_args.push("--ivf".into());
    command_args.push("-o".into());
    command_args.push(args.output.as_os_str().to_os_string());
    if let Some(path) = &args.frame_time_csv {
        command_args.push("--frame-time-csv".into());
        command_args.push(path.as_os_str().to_os_string());
    }
    command_args.push(args.input.as_os_str().to_os_string());
    command_args
}

fn run_encoder(
    args: &Args,
    timeout: std::time::Duration,
) -> Result<std::process::ExitStatus, String> {
    rvp9_bench::process::run_status(
        Command::new(&args.encoder).args(encoder_args(args)),
        "realtime encoder",
        timeout,
    )
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if help_requested(&argv) {
        print!("{}", usage());
        return;
    }
    let args = parse_args();

    println!("encoder:         {}", args.encoder.display());
    println!("input:           {}", args.input.display());
    println!("cpu-used:        {}", args.cpu_used);
    println!("target-bitrate:  {} kbps", args.target_bitrate_kbps);
    println!("end-usage:       {}", args.end_usage);
    println!("threads:          {}", args.threads);
    println!("tile-columns:     {}", args.tile_columns);
    println!("aq-mode:          {}", args.aq_mode);
    println!("noise-sensitivity: {}", args.noise_sensitivity);
    println!("kf-max-dist:      {}", args.kf_max_dist);
    println!("lag-in-frames:    0");
    println!("tune-content:     default");
    if let Some(layers) = args.temporal_layers {
        println!("temporal-layers:  {layers}");
    }
    if let Some(layers) = args.spatial_layers {
        println!("spatial-layers:   {layers}");
        println!("inter-layer-pred: {}", args.inter_layer_pred);
    }
    if let Some(limit) = args.limit {
        println!("limit:            {limit}");
    }
    println!("budget-ms:       {:.3}", args.budget_ms);
    println!("output:          {}", args.output.display());
    if let Some(path) = &args.frame_time_csv {
        println!("frame-time-csv:  {}", path.display());
    }
    if let Some(budget) = args.p99_budget_ms {
        println!("p99-budget-ms:   {budget:.3}");
    }
    println!();

    let start = Instant::now();
    let status = run_encoder(&args, rvp9_bench::process::DEFAULT_TIMEOUT).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1)
    });
    let elapsed = start.elapsed();

    if !status.success() {
        eprintln!("encoder exited with {status}");
        std::process::exit(1);
    }

    let frames = count_ivf_frames(&args.output);
    if frames == 0 {
        eprintln!("no frames in output IVF — encoder may have produced no data");
        std::process::exit(2);
    }

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_ms_per_frame = total_ms / f64::from(frames);
    let effective_fps = 1000.0 / avg_ms_per_frame;

    let timing_stats = if let Some(path) = &args.frame_time_csv {
        match read_frame_time_csv(path, frames as usize) {
            Ok(stats) => Some(stats),
            Err(error) => {
                eprintln!("invalid frame timing CSV: {error}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };

    println!("frames:          {frames}");
    println!("total time:      {:.2}s", elapsed.as_secs_f64());
    println!("avg ms/frame:    {avg_ms_per_frame:.2}");
    println!("effective fps:   {effective_fps:.1}");
    println!("budget:          {:.3} ms/frame", args.budget_ms);
    if let Some(stats) = timing_stats {
        println!("timing samples:  {}", stats.n);
        println!("encode p50:      {:.2} ms", stats.p50);
        println!("encode p90:      {:.2} ms", stats.p90);
        println!("encode p99:      {:.2} ms", stats.p99);
    }

    if avg_ms_per_frame <= args.budget_ms {
        let headroom_pct = (args.budget_ms - avg_ms_per_frame) / args.budget_ms * 100.0;
        println!("\nPASS: under budget by {headroom_pct:.1}%");
    } else {
        let over_pct = (avg_ms_per_frame - args.budget_ms) / args.budget_ms * 100.0;
        println!("\nFAIL: over budget by {over_pct:.1}%");
        std::process::exit(3);
    }

    if let (Some(p99_budget_ms), Some(stats)) = (args.p99_budget_ms, timing_stats) {
        if stats.p99 > p99_budget_ms {
            let over_pct = (stats.p99 - p99_budget_ms) / p99_budget_ms * 100.0;
            println!(
                "FAIL: p99 encode time {:.2} ms exceeds budget {:.3} ms by {over_pct:.1}%",
                stats.p99, p99_budget_ms
            );
            std::process::exit(4);
        }
        let headroom_pct = (p99_budget_ms - stats.p99) / p99_budget_ms * 100.0;
        println!("PASS: p99 under budget by {headroom_pct:.1}%");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn external_encoder_timeout() {
        let fixture = crate::process_test_support::Fixture::new();
        let mut args = test_args();
        args.encoder = fixture.tool.clone();
        let error = run_encoder(&args, crate::process_test_support::DEADLINE).unwrap_err();
        assert!(
            error.contains("realtime encoder") && error.contains("timed out"),
            "{error}"
        );
        assert!(error.contains("fake stderr diagnostic"), "{error}");
    }

    fn csv(rows: &[(usize, u8, u128)]) -> String {
        let mut text = format!("{FRAME_TIME_CSV_HEADER}\n");
        for (index, temporal_layer_id, encode_ns) in rows {
            text.push_str(&format!("{index},{temporal_layer_id},{encode_ns}\n"));
        }
        text
    }

    #[test]
    fn explicit_help_is_a_clean_capability_probe() {
        let argv =
            |items: &[&str]| -> Vec<String> { items.iter().map(ToString::to_string).collect() };
        // Both spellings answer on an argv with no `--input`, which is exactly
        // what a capability probe supplies.
        assert!(help_requested(&argv(&["realtime", "--help"])));
        assert!(help_requested(&argv(&["realtime", "-h"])));
        assert!(!help_requested(&argv(&["realtime", "--input", "clip.y4m"])));

        let help = usage();
        assert!(help.starts_with("Usage: realtime --input CLIP.y4m"));
        for flag in [
            "--encoder",
            "--output",
            "--cpu-used",
            "--target-bitrate",
            "--end-usage",
            "--threads",
            "--tile-columns",
            "--temporal-layers",
            "--spatial-layers",
            "--inter-layer-pred",
            "--aq-mode",
            "--noise-sensitivity",
            "--limit",
            "--kf-max-dist",
            "--budget-ms",
            "--frame-time-csv",
            "--p99-budget-ms",
        ] {
            assert!(help.contains(flag), "usage must advertise {flag}");
        }
    }

    #[test]
    fn frame_time_csv_uses_nearest_rank_percentiles() {
        let rows: Vec<(usize, u8, u128)> = (0..100)
            .map(|index| (index, 0, (index as u128 + 1) * 1_000_000))
            .collect();
        let stats = parse_frame_time_csv(&csv(&rows), rows.len()).expect("valid timing CSV");
        assert_eq!(stats.n, 100);
        assert_eq!(stats.p50, 50.0);
        assert_eq!(stats.p90, 90.0);
        assert_eq!(stats.p99, 99.0);
    }

    #[test]
    fn frame_time_csv_rejects_empty_malformed_and_wrong_count() {
        assert!(parse_frame_time_csv("", 1).unwrap_err().contains("empty"));
        assert!(
            parse_frame_time_csv("wrong\n0,0,1\n", 1)
                .unwrap_err()
                .contains("header")
        );
        assert!(
            parse_frame_time_csv(&format!("{FRAME_TIME_CSV_HEADER}\n"), 1)
                .unwrap_err()
                .contains("no data rows")
        );
        assert!(
            parse_frame_time_csv(&csv(&[(1, 0, 1)]), 1)
                .unwrap_err()
                .contains("expected 0")
        );
        assert!(
            parse_frame_time_csv(&csv(&[(0, 0, 1)]), 2)
                .unwrap_err()
                .contains("IVF contains 2 frames")
        );
        assert!(
            parse_frame_time_csv(&format!("{FRAME_TIME_CSV_HEADER}\n0,0,nope\n"), 1)
                .unwrap_err()
                .contains("invalid encode_ns")
        );
        assert!(
            parse_frame_time_csv(&format!("{FRAME_TIME_CSV_HEADER}\n0,0,0\n"), 1)
                .unwrap_err()
                .contains("encode_ns=0")
        );
        assert!(
            parse_frame_time_csv(&format!("{FRAME_TIME_CSV_HEADER}\n0,3,1\n"), 1)
                .unwrap_err()
                .contains("expected 0..=2")
        );
    }

    fn test_args() -> Args {
        Args {
            encoder: PathBuf::from("rvp9enc"),
            input: PathBuf::from("input.y4m"),
            cpu_used: 7,
            target_bitrate_kbps: 2000,
            budget_ms: 33.333,
            output: PathBuf::from("output.ivf"),
            end_usage: "cbr".to_string(),
            threads: 1,
            tile_columns: 0,
            temporal_layers: None,
            spatial_layers: None,
            inter_layer_pred: 0,
            aq_mode: 0,
            noise_sensitivity: 0,
            limit: None,
            kf_max_dist: 60,
            frame_time_csv: None,
            p99_budget_ms: None,
        }
    }

    #[test]
    fn parses_and_forwards_realtime_matrix_options() {
        let argv = vec![
            "realtime".to_string(),
            "--input=input.y4m".to_string(),
            "--cpu-used=6".to_string(),
            "--end-usage=vbr".to_string(),
            "--threads=4".to_string(),
            "--tile-columns=2".to_string(),
            "--temporal-layers=3".to_string(),
            "--aq-mode=3".to_string(),
            "--noise-sensitivity=2".to_string(),
            "--limit=12".to_string(),
            "--kf-max-dist=48".to_string(),
        ];
        let args = parse_args_from(&argv);
        assert_eq!(args.cpu_used, 6);
        assert_eq!(args.end_usage, "vbr");
        assert_eq!(args.threads, 4);
        assert_eq!(args.tile_columns, 2);
        assert_eq!(args.temporal_layers, Some(3));
        assert_eq!(args.aq_mode, 3);
        assert_eq!(args.noise_sensitivity, 2);
        assert_eq!(args.limit, Some(12));
        assert_eq!(args.kf_max_dist, 48);

        let command_args = encoder_args(&args);
        let command_args: Vec<String> = command_args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        for expected in [
            "--cpu-used=6",
            "--end-usage=vbr",
            "--threads=4",
            "--tile-columns=2",
            "--temporal-layers=3",
            "--aq-mode=3",
            "--noise-sensitivity=2",
            "--limit=12",
            "--kf-max-dist=48",
            "--lag-in-frames=0",
            "--tune-content=default",
        ] {
            assert!(command_args.iter().any(|arg| arg == expected), "{expected}");
        }
    }

    #[test]
    fn command_defaults_preserve_single_layer_cbr_shape() {
        let args = test_args();
        let command_args = encoder_args(&args);
        let command_args: Vec<String> = command_args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        for expected in [
            "--end-usage=cbr",
            "--threads=1",
            "--tile-columns=0",
            "--aq-mode=0",
            "--noise-sensitivity=0",
            "--kf-max-dist=60",
            "--lag-in-frames=0",
            "--tune-content=default",
        ] {
            assert!(command_args.iter().any(|arg| arg == expected), "{expected}");
        }
        assert!(
            !command_args
                .iter()
                .any(|arg| arg.starts_with("--temporal-layers="))
        );
    }

    #[test]
    fn parses_and_forwards_spatial_svc_controls() {
        let argv = vec![
            "realtime".to_string(),
            "--input=input.y4m".to_string(),
            "--spatial-layers=3".to_string(),
            "--inter-layer-pred=0".to_string(),
        ];
        let args = parse_args_from(&argv);
        let command_args = encoder_args(&args);
        let command_args: Vec<String> = command_args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();

        for expected in ["--spatial-layers=3", "--inter-layer-pred=0"] {
            assert!(command_args.iter().any(|arg| arg == expected), "{expected}");
        }
    }

    #[test]
    #[should_panic(expected = "--temporal-layers and --spatial-layers cannot be combined")]
    fn rejects_combined_temporal_and_spatial_svc() {
        parse_args_from(&[
            "realtime".to_string(),
            "--input=input.y4m".to_string(),
            "--temporal-layers=3".to_string(),
            "--spatial-layers=3".to_string(),
        ]);
    }
}

#[cfg(all(test, unix))]
#[path = "../process_test_support.rs"]
mod process_test_support;
