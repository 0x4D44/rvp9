//! T-5 per-tile-dispatch profiling driver.
//!
//! Sets `RVP9_PROFILE_TILES_OUT=<--out>`, constructs
//! [`rvp9_decoder::Decoder::with_threads(--threads)`], and replays the
//! `--fixture` file in a loop until at least `--min-timed` post-warmup
//! profiling rows have been emitted; repeats for `--runs` independent decoder
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
//! Requires x86_64 and the decoder's `profile-tiles` feature (also enabled
//! by `rvp9-bench/profile-tiles`). Runs and minimum timed rows must be
//! positive. CSV output includes warmup rows; the timed summary excludes them.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rvp9_decoder::{DecodeError, Decoder, tile_profile_cpu_mhz, tile_profile_output_status};
use rvp9_harness::container::read_packets;

const DEFAULT_FIXTURE_REL: &str = "tests/conformance/vectors/vp90-2-08-tile_1x4.webm";
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

fn parse_args() -> Result<Args, String> {
    parse_argv(&std::env::args().collect::<Vec<_>>())
}

fn parse_argv(argv: &[String]) -> Result<Args, String> {
    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    let numeric = |flag: &str, default: usize, minimum: usize| -> Result<usize, String> {
        let Some(i) = argv.iter().position(|a| a == flag) else {
            return Ok(default);
        };
        let value = argv
            .get(i + 1)
            .ok_or_else(|| format!("{flag} requires an integer"))?;
        let value: usize = value
            .parse()
            .map_err(|_| format!("{flag} requires an integer >= {minimum}"))?;
        if value < minimum {
            return Err(format!("{flag} must be >= {minimum}"));
        }
        Ok(value)
    };
    let root = workspace_root();
    Ok(Args {
        fixture: get("--fixture")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join(DEFAULT_FIXTURE_REL)),
        threads: NonZeroUsize::new(numeric("--threads", DEFAULT_THREADS, 1)?).unwrap(),
        runs: numeric("--runs", DEFAULT_RUNS, 1)?,
        warmup: numeric("--warmup", DEFAULT_WARMUP, 0)?,
        min_timed: numeric("--min-timed", DEFAULT_MIN_TIMED, 1)?,
        out: get("--out")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join(DEFAULT_OUT_REL)),
    })
}

fn profile_rows_written() -> Result<u64, String> {
    let status = tile_profile_output_status().ok_or_else(|| {
        "unsupported tile profiling: requires x86_64 and --features rvp9-decoder/profile-tiles (or rvp9-bench/profile-tiles)".to_owned()
    })?;
    if status.write_failed {
        return Err("tile profile CSV output failed; no valid profiling result".to_owned());
    }
    Ok(status.rows_written)
}

fn drive_one_run(
    fixture: &Path,
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
    let mut timed_rows: usize = 0;
    let mut total_wall_us: f64 = 0.0;
    let mut decoder = Decoder::with_threads(threads);

    'outer: loop {
        let mut produced_any_this_pass = false;
        let timed_before_pass = timed_rows;
        // A pass crossing the warmup boundary may have emitted its only
        // profile row during warmup. Give the next full timed pass a chance.
        let fully_timed_pass = frames_seen >= warmup;
        for (packet_offset, range) in packets.iter().enumerate() {
            let packet = range.slice(&data);
            let parsed = decoder.parse_packet(packet).map_err(|error| {
                format!(
                    "{}: packet {}: VP9 parse error: {error:?}",
                    fixture.display(),
                    packet_offset + 1
                )
            })?;
            for frame in parsed {
                let rows_before = profile_rows_written()?;
                let t0 = Instant::now();
                let result = decoder.decode_frame(&frame);
                let dt = t0.elapsed();
                let rows_after = profile_rows_written()?;
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
                    let emitted = usize::try_from(rows_after - rows_before)
                        .map_err(|_| "profile row count exceeds usize".to_owned())?;
                    timed_rows += emitted;
                    if emitted > 0 {
                        total_wall_us += dt.as_secs_f64() * 1_000_000.0;
                    }
                    if timed_rows >= min_timed {
                        break 'outer;
                    }
                }
            }
        }
        if !produced_any_this_pass {
            return Err("fixture produced no decodable frames or profiling evidence".to_owned());
        }
        if fully_timed_pass && timed_rows == timed_before_pass {
            return Err("fixture pass emitted no timed profile CSV rows; use a supported multi-tile fixture and enough threads".to_owned());
        }
    }
    Ok((timed_rows, total_wall_us))
}

fn main() {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let mhz = tile_profile_cpu_mhz().filter(|mhz| mhz.is_finite() && *mhz > 0.0).unwrap_or_else(|| {
        eprintln!("unsupported tile profiling: requires x86_64 and --features rvp9-decoder/profile-tiles (or rvp9-bench/profile-tiles) with a positive CPU MHz calibration");
        std::process::exit(1);
    });

    println!("decode-profile-tiles (T-5 instrumentation)");
    println!("  fixture:    {}", args.fixture.display());
    println!("  threads:    {}", args.threads.get());
    println!("  runs:       {}", args.runs);
    println!("  warmup:     {} frames/run", args.warmup);
    println!("  min timed:  {} frames/run", args.min_timed);
    println!("  out (CSV):  {}", args.out.display());

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
    unsafe {
        std::env::set_var("RVP9_PROFILE_TILES_OUT", &args.out);
    }

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
                    "[run {run}/{}] timed {n} CSV rows, total {:.1} ms",
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
        "wrote {total_timed} timed CSV rows (warmup rows excluded) across {runs} runs in {:.1} s; mean {:.2} ms/frame",
        elapsed.as_secs_f64(),
        total_wall_us / 1000.0 / total_timed as f64,
        runs = args.runs,
    );
    println!("CSV: {}", args.out.display());
    println!("Note: convert cycles → microseconds with cycles / {mhz:.1} = us");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_harness::container::{IvfHeader, write_ivf};

    #[test]
    fn empty_fixture_is_not_profile_evidence() {
        let path = std::env::temp_dir().join(format!(
            "rvp9-profile-tiles-empty-{}.ivf",
            std::process::id()
        ));
        let mut ivf = Vec::new();
        write_ivf(&mut ivf, &IvfHeader::vp9(16, 16, 30, 1), &[]).unwrap();
        std::fs::write(&path, ivf).unwrap();
        let result = drive_one_run(&path, NonZeroUsize::MIN, 0, 1);
        assert!(
            result.is_err(),
            "empty fixture falsely succeeded: {result:?}"
        );
    }

    #[test]
    fn malformed_packet_fails_profile_run_with_fixture_and_index() {
        let path = std::env::temp_dir().join(format!(
            "rvp9-profile-tiles-malformed-packet-{}.ivf",
            std::process::id()
        ));
        let mut ivf = Vec::new();
        write_ivf(&mut ivf, &IvfHeader::vp9(16, 16, 30, 1), &[&[0u8][..]]).unwrap();
        std::fs::write(&path, ivf).unwrap();

        let err = drive_one_run(&path, NonZeroUsize::MIN, 0, 1).unwrap_err();
        std::fs::remove_file(&path).unwrap();

        assert!(err.contains(&path.display().to_string()), "{err}");
        assert!(err.contains("packet 1"), "{err}");
        assert!(err.contains("VP9 parse error"), "{err}");
    }
}
