//! Decoder throughput benchmark.
//!
//! Drives [`rvp9_decoder::Decoder`] against a curated set of existing
//! VP9 fixtures (the AOM conformance corpus that ships gitignored under
//! `tests/conformance/vectors/`) and records per-frame decode wall-time
//! as median / p95 / p99, in microseconds.
//!
//! Per HLD §G6 + §6.4:
//!   - First [`rvp9_bench::throughput::WARMUP_FRAMES`] frames per
//!     fixture are dropped from the timing window.
//!   - The fixture is decoded in a loop until at least
//!     [`rvp9_bench::throughput::MIN_TIMED_FRAMES`] frames have been
//!     timed (single-clip p99 is meaningless otherwise).
//!   - A CSV row per fixture is appended to
//!     `bench/results/decode-throughput.csv` for trend tracking.
//!
//! The hard real-time gate (720p30 ≤ 33ms/frame median single-core etc.)
//! lands as part of Wave C exit per HLD §6.4. This binary is the
//! *measurement infrastructure*; the soft ratchet lives in the
//! integration test `decode_throughput_ratchet.rs`.
//!
//! Fixture generation (G6.1, resolved Wave L L6): a working libvpx
//! `vpxenc.exe` lives at `build/libvpx/`, so fresh `.ivf` fixtures can
//! be generated from the `testsrc_*.y4m` raw sources (HLD §G6's
//! preferred input) via `scripts/encode-corpus.ps1`, e.g.
//! `pwsh scripts/encode-corpus.ps1 -Src tests/corpus/testsrc_720p30.y4m
//! -Out tests/corpus/fixtures/testsrc_720p30.ivf`. The bench only
//! decodes for timing (no bit-check), so a self-encoded fixture is
//! valid input regardless of bit-exact conformance.
//!
//! # Pinned vs unpinned rows
//!
//! Each CSV row carries an `unpinned` flag. The default is `true`
//! (dev-box run). Pass `--pinned` only on the dedicated Linux CI
//! runner with `taskset`/`cpupower` discipline applied — per HLD §6.4
//! that's the only host where the ratchet baseline is meaningful.
//!
//! Usage:
//!   cargo run --release -p rvp9-bench --bin decode-throughput
//!   cargo run --release -p rvp9-bench --bin decode-throughput -- --pinned
//!
//! Exit code is always 0 on a successful measurement; the ratchet
//! gate is a separate `cargo test` integration test so individual bench
//! invocations don't block CI on transient noise.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rvp9_bench::throughput::{
    self, FixtureResult, MIN_TIMED_FRAMES, WARMUP_FRAMES, build_profile_hash, host_fingerprint,
};

struct Args {
    csv: PathBuf,
    fixtures: Vec<PathBuf>,
    unpinned: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    let has = |flag: &str| -> bool { argv.iter().any(|a| a == flag) };

    // Default CSV is `bench/results/decode-throughput.csv` at the
    // workspace root (two levels up from this crate).
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = crate_root
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or(crate_root);
    let default_csv = workspace_root
        .join("bench")
        .join("results")
        .join("decode-throughput.csv");

    let csv = get("--csv").map(PathBuf::from).unwrap_or(default_csv);

    let vectors = throughput::vectors_dir();
    // `--fixture <path>` overrides the default vector set with a single
    // explicit clip (used to profile non-default corpus clips, e.g.
    // synthetic testsrc, where rvp9's fixed-overhead gap is worst).
    let fixtures: Vec<PathBuf> = if let Some(p) = get("--fixture") {
        vec![PathBuf::from(p)]
    } else {
        throughput::DEFAULT_FIXTURES
            .iter()
            .map(|f| vectors.join(f))
            .collect()
    };

    // Default to unpinned (dev box). Operator on the dedicated CI
    // runner opts in with `--pinned`.
    let unpinned = !has("--pinned");

    Args {
        csv,
        fixtures,
        unpinned,
    }
}

fn main() {
    let args = parse_args();
    let host = host_fingerprint();
    let bph = build_profile_hash();

    println!("decode-throughput benchmark");
    println!("  host:        {host}");
    println!("  pinned:      {}", !args.unpinned);
    println!("  build hash:  {bph}");
    println!("  csv:         {}", args.csv.display());
    println!("  warmup:      {WARMUP_FRAMES} frames/fixture");
    println!("  min timed:   {MIN_TIMED_FRAMES} frames/fixture");
    println!("  fixtures:    {}", args.fixtures.len());
    println!();

    let mut results: Vec<FixtureResult> = Vec::with_capacity(args.fixtures.len());
    let mut had_error = false;

    for path in &args.fixtures {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        eprintln!("[run] {name}");
        match throughput::run_one(path, WARMUP_FRAMES, MIN_TIMED_FRAMES) {
            Ok(r) => {
                println!(
                    "  {fix:<40} {w}x{h:<5}  n={n:>5}  p50={p50:>7.1} us  p95={p95:>7.1} us  p99={p99:>7.1} us",
                    fix = r.fixture,
                    w = r.width,
                    h = r.height,
                    n = r.timed_frames,
                    p50 = r.stats.p50,
                    p95 = r.stats.p95,
                    p99 = r.stats.p99,
                );
                results.push(r);
            }
            Err(e) => {
                eprintln!("  ERROR: {e}");
                had_error = true;
            }
        }
    }

    if results.is_empty() {
        eprintln!("\nno fixtures produced timed samples; nothing to record");
        // Don't exit non-zero — the ratchet test is the gate.
        return;
    }

    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if let Err(e) = throughput::append_csv(
        &args.csv,
        timestamp_unix,
        &host,
        args.unpinned,
        &bph,
        &results,
    ) {
        eprintln!("\nfailed to append CSV: {e}");
        return;
    }

    println!("\nwrote {} row(s) to {}", results.len(), args.csv.display());
    if had_error {
        eprintln!("(one or more fixtures failed; see ERROR lines above)");
    }
}
