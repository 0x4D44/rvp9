//! T-5 per-tile-dispatch profiling driver.
//!
//! Sets `RVP9_PROFILE_TILES_OUT=<--out>`, constructs
//! [`rvp9_decoder::Decoder::with_threads(--threads)`], and replays the
//! `--fixture` file in a loop until at least `--min-timed` post-warmup
//! frames have been timed; repeats for `--runs` independent decoder
//! instances. Each frame appends one CSV row of phase rdtsc-cycles
//! deltas via the decoder's `profile::emit_row` (gated on the
//! `profile-tiles` feature).
//!
//! Usage:
//! ```text
//! cargo run --release --features rvp9-decoder/profile-tiles \
//!   --bin decode-profile-tiles -- \
//!   --fixture tests/conformance/vectors/vp90-2-08-tile_1x4.webm \
//!   --threads 4 --runs 10 --warmup 30 --min-timed 1000 \
//!   --out bench/results/t5-tile-profile.tile_1x4.t4.csv
//! ```
//!
//! Defaults match the HLD §4.2 procedure.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use rvp9_decoder::{DecodeError, Decoder};
use rvp9_harness::container::read_packets;

const DEFAULT_FIXTURE_REL: &str =
    "tests/conformance/vectors/vp90-2-08-tile_1x4.webm";
const DEFAULT_THREADS: usize = 4;
const DEFAULT_RUNS: usize = 10;
const DEFAULT_WARMUP: usize = 30;
const DEFAULT_MIN_TIMED: usize = 1_000;
const DEFAULT_OUT_REL: &str = "bench/results/t5-tile-profile.tile_1x4.t4.csv";

struct Args {
    fixture: PathBuf,
    threads: NonZeroUsize,
    runs: usize,
    warmup: usize,
    min_timed: usize,
    out: PathBuf,
}

fn workspace_root() -> PathBuf {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or(crate_root)
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    let root = workspace_root();
    let fixture = get("--fixture")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join(DEFAULT_FIXTURE_REL));
    let threads: usize = get("--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_THREADS);
    let threads = NonZeroUsize::new(threads).unwrap_or_else(|| {
        eprintln!("--threads must be >= 1");
        std::process::exit(2);
    });
    let runs = get("--runs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_RUNS);
    let warmup = get("--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WARMUP);
    let min_timed = get("--min-timed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MIN_TIMED);
    let out = get("--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join(DEFAULT_OUT_REL));
    Args {
        fixture,
        threads,
        runs,
        warmup,
        min_timed,
        out,
    }
}

#[cfg(all(feature = "profile-tiles", target_arch = "x86_64"))]
fn cpu_mhz() -> f64 {
    rvp9_decoder::profile::estimate_cpu_mhz()
}

#[cfg(not(all(feature = "profile-tiles", target_arch = "x86_64")))]
fn cpu_mhz() -> f64 {
    eprintln!(
        "WARNING: built without the `profile-tiles` feature; \
         no CSV will be emitted. Rebuild with \
         `--features rvp9-decoder/profile-tiles`."
    );
    0.0
}

fn drive_one_run(
    fixture: &PathBuf,
    threads: NonZeroUsize,
    warmup: usize,
    min_timed: usize,
) -> Result<(usize, f64), String> {
    if !fixture.exists() {
        return Err(format!("missing fixture: {}", fixture.display()));
    }
    let (data, packets) =
        read_packets(fixture).map_err(|e| format!("container parse error: {e}"))?;

    let mut frames_seen: usize = 0;
    let mut timed_frames: usize = 0;
    let mut total_wall_us: f64 = 0.0;
    let mut decoder = Decoder::with_threads(threads);

    'outer: loop {
        let mut produced_any_this_pass = false;
        for range in &packets {
            let packet = range.slice(&data);
            let parsed = match decoder.parse_packet(packet) {
                Ok(p) => p,
                Err(_) => continue,
            };
            for frame in parsed {
                let t0 = Instant::now();
                let result = decoder.decode_frame(&frame);
                let dt = t0.elapsed();
                match result {
                    Ok(_buf) => {}
                    Err(DecodeError::Unimplemented { .. }) => continue,
                    Err(e) => {
                        return Err(format!("decode error: {e:?}"));
                    }
                }
                produced_any_this_pass = true;
                frames_seen += 1;
                if frames_seen > warmup {
                    timed_frames += 1;
                    total_wall_us += dt.as_secs_f64() * 1_000_000.0;
                    if timed_frames >= min_timed {
                        break 'outer;
                    }
                }
            }
        }
        if !produced_any_this_pass {
            break;
        }
    }
    Ok((timed_frames, total_wall_us))
}

fn main() {
    let args = parse_args();

    println!("decode-profile-tiles (T-5 instrumentation)");
    println!("  fixture:    {}", args.fixture.display());
    println!("  threads:    {}", args.threads.get());
    println!("  runs:       {}", args.runs);
    println!("  warmup:     {} frames/run", args.warmup);
    println!("  min timed:  {} frames/run", args.min_timed);
    println!("  out (CSV):  {}", args.out.display());

    // Estimate CPU MHz once (also caches it inside the decoder for
    // any future callers in the same process).
    let mhz = cpu_mhz();
    println!("  cpu MHz:    {mhz:.1}");

    // Ensure parent dir exists, then expose path to the decoder.
    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!("failed to create output dir {}: {e}", parent.display());
        std::process::exit(1);
    }
    // SAFETY: single-threaded at this point; no other thread reads
    // the env. The decoder's profile sink picks it up via `OnceLock`
    // on first emit_row call.
    // SAFETY: see the comment above — env mutation is single-threaded
    // at process start, before the decoder spawns workers.
    unsafe { std::env::set_var("RVP9_PROFILE_TILES_OUT", &args.out); }

    let total_start = Instant::now();
    let mut total_timed: usize = 0;
    let mut total_wall_us: f64 = 0.0;
    for run in 1..=args.runs {
        eprintln!("[run {run}/{}] starting", args.runs);
        match drive_one_run(&args.fixture, args.threads, args.warmup, args.min_timed) {
            Ok((n, wall)) => {
                total_timed += n;
                total_wall_us += wall;
                eprintln!(
                    "[run {run}/{}] timed {n} frames, total {:.1} ms",
                    args.runs,
                    wall / 1000.0
                );
            }
            Err(e) => {
                eprintln!("[run {run}/{}] ERROR: {e}", args.runs);
                std::process::exit(1);
            }
        }
    }
    let elapsed = total_start.elapsed();
    println!(
        "wrote {total_timed} CSV rows across {runs} runs in {:.1} s; mean {:.2} ms/frame",
        elapsed.as_secs_f64(),
        if total_timed == 0 {
            0.0
        } else {
            total_wall_us / 1000.0 / total_timed as f64
        },
        runs = args.runs,
    );
    println!("CSV: {}", args.out.display());
    println!(
        "Note: convert cycles → microseconds with cycles / {mhz:.1} = us"
    );
}
