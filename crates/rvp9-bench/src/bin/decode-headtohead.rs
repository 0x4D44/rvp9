//! Decoder head-to-head benchmark — rvp9dec side.
//!
//! Drives the `(fixture, threads)` matrix from
//! `wrk_docs/Track_Q/Q.1.A_e2e_perf_HLD.md` §5 through
//! [`rvp9_bench::throughput::run_one_threaded`] and emits a TSV result
//! file that the PowerShell orchestrator (`scripts/bench-headtohead.ps1`)
//! joins with `vpxdec --noblit --summary -t N` wall-times to produce the
//! committed ledger at `bench/results/decode-headtohead.md` + `.csv`.
//!
//! # Independence from Q.1.B
//!
//! Per HLD §5 + §11.3: this binary calls
//! [`rvp9_decoder::Decoder::with_threads`] directly through the bench
//! library; it does NOT shell out to the CLI subprocess. Q.1.A is not
//! blocked on Q.1.B.
//!
//! # Output schema (TSV — intermediate, schema v2)
//!
//! This TSV is the rvp9-side intermediate that the PowerShell
//! orchestrator consumes; the final committed ledger
//! (`bench/results/decode-headtohead.csv`, schema v3) is produced by
//! that orchestrator after joining with vpxdec wall-times. The two
//! schemas are versioned independently.
//!
//! Header line: `# rvp9-headtohead schema=2 host=<fp> build=<hash>
//!   run_index=<N>`. The `run_index` field is 1-based and is propagated
//! to every data row so the orchestrator can stamp it into the CSV
//! when running N-run sweeps for statistical comparison (Task C).
//!
//! Tab-separated columns:
//!
//! ```text
//! label\tfixture_path\tthreads\twidth\theight\ttimed_frames\
//!   \tp50_us\tmean_us\tp95_us\tp99_us\trun_index
//! ```
//!
//! Tabs (not commas) so fixture paths with spaces survive the join
//! step in PowerShell.
//!
//! # CLI
//!
//! ```text
//! decode-headtohead [--out <tsv>] [--warmup N] [--min-timed N]
//!                   [--run-index N]
//!                   --row <label>=<path>:<threads> [--row ...]
//! ```
//!
//! `--run-index` defaults to 1 (single-run behaviour identical to
//! schema v1). With no `--row` flags the built-in default matrix runs
//! (HLD §5).

use std::num::NonZeroUsize;
use std::path::PathBuf;

use rvp9_bench::throughput::{
    self, FixtureResult, MIN_TIMED_FRAMES, ThroughputError, WARMUP_FRAMES, build_profile_hash,
    host_fingerprint,
};

/// One row of the HLD §5 matrix: a fixture to decode with a fixed thread
/// count, tagged with a short stable label used as the join key with the
/// vpxdec side.
#[derive(Debug, Clone)]
struct Row {
    /// Short stable label — joins with the vpxdec result. E.g.
    /// `quantizer-00`, `tile_1x4`, `testsrc_1080p30`.
    label: String,
    /// Absolute or workspace-relative path to the fixture.
    path: PathBuf,
    /// Tile/worker count for [`rvp9_decoder::Decoder::with_threads`].
    threads: NonZeroUsize,
}

struct Args {
    out: PathBuf,
    warmup: usize,
    min_timed: usize,
    /// 1-based run index. Stamped into the TSV header and into every
    /// row so N-run sweeps can attribute samples back to their iteration
    /// downstream (CSV schema v3).
    run_index: u32,
    rows: Vec<Row>,
}

fn workspace_root() -> PathBuf {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or(crate_root)
}

/// Default matrix per HLD §5. Fixture availability is checked at run
/// time; missing rows are reported but don't abort the rest of the run.
fn default_matrix() -> Vec<Row> {
    let root = workspace_root();
    let vectors = root.join("tests").join("conformance").join("vectors");
    let corpus_fix = root.join("tests").join("corpus").join("fixtures");

    let nz = |n: usize| NonZeroUsize::new(n).expect("hard-coded matrix uses non-zero counts");

    vec![
        // A: single-tile control (HLD §2 row A) — threads=1 only.
        Row {
            label: "quantizer-00".into(),
            path: vectors.join("vp90-2-00-quantizer-00.webm"),
            threads: nz(1),
        },
        // B: multi-tile control (HLD §2 row B) — sweep {1, 4}.
        Row {
            label: "tile_1x4".into(),
            path: vectors.join("vp90-2-08-tile_1x4.webm"),
            threads: nz(1),
        },
        Row {
            label: "tile_1x4".into(),
            path: vectors.join("vp90-2-08-tile_1x4.webm"),
            threads: nz(4),
        },
        // C: 720p30 (HLD §2 row C) — single-core only.
        Row {
            label: "testsrc_720p30".into(),
            path: corpus_fix.join("testsrc_720p30.ivf"),
            threads: nz(1),
        },
        // D: 1080p30 (HLD §2 row D) — sweep {2, 4}.
        Row {
            label: "testsrc_1080p30".into(),
            path: corpus_fix.join("testsrc_1080p30.ivf"),
            threads: nz(2),
        },
        Row {
            label: "testsrc_1080p30".into(),
            path: corpus_fix.join("testsrc_1080p30.ivf"),
            threads: nz(4),
        },
        // E: 1080p60 (HLD §2 row E) — sweep {4, 8}.
        Row {
            label: "testsrc_1080p60".into(),
            path: corpus_fix.join("testsrc_1080p60.ivf"),
            threads: nz(4),
        },
        Row {
            label: "testsrc_1080p60".into(),
            path: corpus_fix.join("testsrc_1080p60.ivf"),
            threads: nz(8),
        },
        // F: Profile-1 8-bit non-default (HLD §2 row F) — threads=1.
        Row {
            label: "foreman-fp1".into(),
            path: corpus_fix.join("foreman-fp1.ivf"),
            threads: nz(1),
        },
    ]
}

fn parse_row_spec(spec: &str) -> Result<Row, String> {
    // Format: `<label>=<path>:<threads>`. We split from the RIGHT so
    // Windows-style paths with colons (`D:\...`) survive.
    let (left, threads_str) = spec
        .rsplit_once(':')
        .ok_or_else(|| format!("--row spec missing `:<threads>` suffix: `{spec}`"))?;
    let (label, path_str) = left
        .split_once('=')
        .ok_or_else(|| format!("--row spec missing `<label>=` prefix: `{spec}`"))?;
    let threads: usize = threads_str
        .parse()
        .map_err(|e| format!("--row threads `{threads_str}` not a positive integer: {e}"))?;
    let threads = NonZeroUsize::new(threads)
        .ok_or_else(|| format!("--row threads must be >= 1, got {threads_str}"))?;
    Ok(Row {
        label: label.to_string(),
        path: PathBuf::from(path_str),
        threads,
    })
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    let get_all = |flag: &str| -> Vec<String> {
        let mut out = Vec::new();
        for (i, a) in argv.iter().enumerate() {
            if a == flag
                && let Some(v) = argv.get(i + 1)
            {
                out.push(v.clone());
            }
        }
        out
    };

    let default_out = workspace_root()
        .join("bench")
        .join("results")
        .join("decode-headtohead.rvp9.tsv");

    let out = get("--out").map(PathBuf::from).unwrap_or(default_out);
    let warmup = get("--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(WARMUP_FRAMES);
    let min_timed = get("--min-timed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(MIN_TIMED_FRAMES);
    let run_index: u32 = get("--run-index").and_then(|s| s.parse().ok()).unwrap_or(1);
    if run_index == 0 {
        eprintln!("error: --run-index must be >= 1 (1-based)");
        std::process::exit(2);
    }

    let row_specs = get_all("--row");
    let rows: Vec<Row> = if row_specs.is_empty() {
        default_matrix()
    } else {
        let mut parsed = Vec::with_capacity(row_specs.len());
        for spec in &row_specs {
            match parse_row_spec(spec) {
                Ok(r) => parsed.push(r),
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(2);
                }
            }
        }
        parsed
    };

    Args {
        out,
        warmup,
        min_timed,
        run_index,
        rows,
    }
}

fn write_tsv_header(buf: &mut String, host: &str, build_hash: &str, run_index: u32) {
    buf.push_str(&format!(
        "# rvp9-headtohead schema=2 host={host} build={build_hash} run_index={run_index}\n"
    ));
    buf.push_str(
        "label\tfixture_path\tthreads\twidth\theight\ttimed_frames\tp50_us\tmean_us\tp95_us\tp99_us\trun_index\n",
    );
}

fn write_tsv_row(buf: &mut String, row: &Row, r: &FixtureResult, run_index: u32) {
    buf.push_str(&format!(
        "{label}\t{path}\t{th}\t{w}\t{h}\t{n}\t{p50:.3}\t{mean:.3}\t{p95:.3}\t{p99:.3}\t{ri}\n",
        label = row.label,
        path = row.path.display(),
        th = row.threads.get(),
        w = r.width,
        h = r.height,
        n = r.timed_frames,
        p50 = r.stats.p50,
        mean = r.stats.mean,
        p95 = r.stats.p95,
        p99 = r.stats.p99,
        ri = run_index,
    ));
}

fn main() {
    let args = parse_args();
    let host = host_fingerprint();
    let bph = build_profile_hash();

    println!("decode-headtohead (rvp9 side)");
    println!("  host:       {host}");
    println!("  build hash: {bph}");
    println!("  out:        {}", args.out.display());
    println!("  warmup:     {} frames/row", args.warmup);
    println!("  min timed:  {} frames/row", args.min_timed);
    println!("  run index:  {}", args.run_index);
    println!("  rows:       {}", args.rows.len());
    println!();

    let mut buf = String::new();
    write_tsv_header(&mut buf, &host, &bph, args.run_index);

    let mut had_error = false;
    let mut measured: usize = 0;

    for row in &args.rows {
        eprintln!(
            "[run] {label} (t={th}) — {p}",
            label = row.label,
            th = row.threads.get(),
            p = row.path.display(),
        );
        match throughput::run_one_threaded(&row.path, row.threads, args.warmup, args.min_timed) {
            Ok(r) => {
                println!(
                    "  {label:<24} t={th}  {w}x{h:<5}  n={n:>5}  \
                     p50={p50:>8.1}us  mean={mean:>8.1}us  p95={p95:>8.1}us  p99={p99:>8.1}us",
                    label = row.label,
                    th = row.threads.get(),
                    w = r.width,
                    h = r.height,
                    n = r.timed_frames,
                    p50 = r.stats.p50,
                    mean = r.stats.mean,
                    p95 = r.stats.p95,
                    p99 = r.stats.p99,
                );
                write_tsv_row(&mut buf, row, &r, args.run_index);
                measured += 1;
            }
            Err(e) => {
                eprintln!("  ERROR: {e}");
                had_error = true;
                if let ThroughputError::MissingFixture(p) = &e {
                    eprintln!(
                        "  hint: encode the fixture first via scripts/encode-corpus.ps1 \
                         (HLD §2). Missing: {}",
                        p.display(),
                    );
                }
            }
        }
    }

    if let Some(parent) = args.out.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!("\nfailed to create output dir {}: {e}", parent.display());
        std::process::exit(1);
    }
    if let Err(e) = std::fs::write(&args.out, &buf) {
        eprintln!("\nfailed to write {}: {e}", args.out.display());
        std::process::exit(1);
    }
    println!(
        "\nwrote {} measured row(s) to {}",
        measured,
        args.out.display(),
    );
    if had_error {
        // Surface the error count but don't fail the process — the
        // orchestrator may still want to join the partial result set
        // with vpxdec's partial run.
        eprintln!("(one or more rows failed; see ERROR lines above)");
    }
}
