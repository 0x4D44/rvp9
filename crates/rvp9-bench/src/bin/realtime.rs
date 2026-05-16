//! Real-time encoding latency benchmark.
//!
//! Drives a configurable encoder binary (defaults to libvpx vpxenc) against a
//! YUV4MPEG (.y4m) input file at a target speed setting and reports aggregate
//! per-frame encode wall-time: total time, frame count, average ms/frame.
//!
//! In Stage 0 this runs `libvpx vpxenc` as our baseline and validates that
//! the harness works. In Stage 4+, the same binary will be re-pointed at
//! `rvp9enc` to gate real-time compliance.
//!
//! Per-frame stats (p50/p90/p99) require per-frame instrumentation in the
//! encoder, which neither vpxenc nor (yet) rvp9enc exposes. For Stage 0 we
//! report aggregate ms/frame from total wall-time divided by frame count;
//! per-frame stats land when rvp9enc grows that hook in Stage 4a.
//!
//! Usage:
//!   cargo run --release --bin realtime -- \
//!       --encoder D:/language/libvpx-stock/vpxenc.exe \
//!       --input   tests/corpus/parkrun_720p30.y4m \
//!       --cpu-used 7 \
//!       --target-bitrate 2000 \
//!       --budget-ms 33

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use rvp9_bench::count_ivf_frames;

struct Args {
    encoder: PathBuf,
    input: PathBuf,
    cpu_used: u32,
    target_bitrate_kbps: u32,
    budget_ms: f64,
    output: PathBuf,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
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
    let target_bitrate_kbps = get("--target-bitrate")
        .map(|s| s.parse::<u32>().expect("--target-bitrate must be u32"))
        .unwrap_or(2000);
    let budget_ms = get("--budget-ms")
        .map(|s| s.parse::<f64>().expect("--budget-ms must be f64"))
        .unwrap_or(33.333);
    let output = get("--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rvp9_realtime_bench.ivf"));

    Args {
        encoder,
        input,
        cpu_used,
        target_bitrate_kbps,
        budget_ms,
        output,
    }
}

fn main() {
    let args = parse_args();

    println!("encoder:         {}", args.encoder.display());
    println!("input:           {}", args.input.display());
    println!("cpu-used:        {}", args.cpu_used);
    println!("target-bitrate:  {} kbps", args.target_bitrate_kbps);
    println!("budget-ms:       {:.3}", args.budget_ms);
    println!("output:          {}", args.output.display());
    println!();

    let start = Instant::now();
    let status = Command::new(&args.encoder)
        .args([
            "--codec=vp9",
            "--passes=1",
            "--rt",
            &format!("--cpu-used={}", args.cpu_used),
            "--end-usage=cbr",
            &format!("--target-bitrate={}", args.target_bitrate_kbps),
            "--lag-in-frames=0",
            "--ivf",
            "-o",
        ])
        .arg(&args.output)
        .arg(&args.input)
        .status()
        .expect("failed to invoke encoder");
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

    println!("frames:          {frames}");
    println!("total time:      {:.2}s", elapsed.as_secs_f64());
    println!("avg ms/frame:    {avg_ms_per_frame:.2}");
    println!("effective fps:   {effective_fps:.1}");
    println!("budget:          {:.3} ms/frame", args.budget_ms);

    if avg_ms_per_frame <= args.budget_ms {
        let headroom_pct = (args.budget_ms - avg_ms_per_frame) / args.budget_ms * 100.0;
        println!("\nPASS: under budget by {headroom_pct:.1}%");
    } else {
        let over_pct = (avg_ms_per_frame - args.budget_ms) / args.budget_ms * 100.0;
        println!("\nFAIL: over budget by {over_pct:.1}%");
        std::process::exit(3);
    }
}
