//! `bench-compare`: pairwise statistical comparison of two head-to-head
//! bench sweeps (Task C, statistical bench framework).
//!
//! Reads `bench/results/decode-headtohead.csv` (schema v3), filters to
//! rows whose `timestamp` matches `--baseline` or `--candidate`, groups
//! by `(fixture, threads)`, and reports per-group:
//!
//! - median `rvp9_p50_us` across the N runs of each side,
//! - delta percent (candidate vs baseline),
//! - Mann-Whitney U statistic + two-sided p-value
//!   (exact for n ≤ ~10/side; normal approx beyond),
//! - gate verdict:
//!     - **WIN**        if p < 0.05 and delta ≤ -5%
//!     - **REGRESSION** if p < 0.05 and delta ≥ +5%
//!     - **NOISE**      otherwise
//!
//! Output: a Markdown table to stdout AND
//! `bench/results/compare-<baseline>-vs-<candidate>.md`.
//!
//! # CLI
//!
//! ```text
//! bench-compare --baseline <ts1> --candidate <ts2> [--csv <path>]
//! ```
//!
//! Both timestamps are Unix seconds (the `timestamp` column in the
//! CSV). When the same `bench-headtohead.ps1 -Runs N` invocation
//! produces all N rows with the same timestamp, the filter cleanly
//! groups one sweep.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rvp9_bench::mwu::mann_whitney_u;

/// One CSV row from `decode-headtohead.csv` after schema-v3 parsing,
/// projected down to just the columns this tool needs.
#[derive(Debug, Clone)]
struct CompareRow {
    timestamp: u64,
    run_index: u32,
    fixture: String,
    threads: u32,
    rvp9_p50_us: f64,
}

struct Args {
    baseline_ts: u64,
    candidate_ts: u64,
    csv_path: PathBuf,
    out_path: Option<PathBuf>,
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
    let baseline_ts: u64 = match get("--baseline").and_then(|s| s.parse().ok()) {
        Some(v) => v,
        None => {
            eprintln!("error: --baseline <unix-timestamp> required");
            std::process::exit(2);
        }
    };
    let candidate_ts: u64 = match get("--candidate").and_then(|s| s.parse().ok()) {
        Some(v) => v,
        None => {
            eprintln!("error: --candidate <unix-timestamp> required");
            std::process::exit(2);
        }
    };
    let default_csv = workspace_root()
        .join("bench")
        .join("results")
        .join("decode-headtohead.csv");
    let csv_path = get("--csv").map(PathBuf::from).unwrap_or(default_csv);
    let out_path = get("--out").map(PathBuf::from);

    Args {
        baseline_ts,
        candidate_ts,
        csv_path,
        out_path,
    }
}

/// Schema-v3 column layout (must match
/// `scripts/bench-headtohead.ps1` and the writer in this PR).
const COL_SCHEMA_VERSION: usize = 0;
const COL_TIMESTAMP: usize = 1;
const COL_RUN_INDEX: usize = 6;
const COL_FIXTURE: usize = 7;
const COL_THREADS: usize = 8;
const COL_RVP9_P50_US: usize = 12;
const SCHEMA_V3_MIN_COLS: usize = 13;

fn parse_csv(path: &Path) -> Result<Vec<CompareRow>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        if lineno == 0 {
            // Header.
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        let line_number = lineno + 1;
        let schema = cols[COL_SCHEMA_VERSION].parse::<u32>().map_err(|e| {
            format!(
                "{}:{line_number}: invalid schema version: {e}",
                path.display()
            )
        })?;
        if schema != 3 {
            // Unknown schemas are intentionally ignored. Their column
            // layouts are not ours to validate.
            continue;
        }
        if cols.len() < SCHEMA_V3_MIN_COLS {
            return Err(format!(
                "{}:{line_number}: schema-v3 row has {} columns; expected at least {SCHEMA_V3_MIN_COLS}",
                path.display(),
                cols.len()
            ));
        }
        let timestamp = cols[COL_TIMESTAMP].parse::<u64>().map_err(|e| {
            format!(
                "{}:{line_number}: invalid schema-v3 timestamp: {e}",
                path.display()
            )
        })?;
        let run_index = cols[COL_RUN_INDEX].parse::<u32>().map_err(|e| {
            format!(
                "{}:{line_number}: invalid schema-v3 run index: {e}",
                path.display()
            )
        })?;
        let fixture = cols[COL_FIXTURE].to_string();
        if fixture.is_empty() {
            return Err(format!(
                "{}:{line_number}: schema-v3 fixture is empty",
                path.display()
            ));
        }
        let threads = cols[COL_THREADS].parse::<u32>().map_err(|e| {
            format!(
                "{}:{line_number}: invalid schema-v3 thread count: {e}",
                path.display()
            )
        })?;
        let rvp9_p50_us = cols[COL_RVP9_P50_US].parse::<f64>().map_err(|e| {
            format!(
                "{}:{line_number}: invalid schema-v3 rvp9 p50: {e}",
                path.display()
            )
        })?;
        if !rvp9_p50_us.is_finite() {
            return Err(format!(
                "{}:{line_number}: schema-v3 rvp9 p50 must be finite",
                path.display()
            ));
        }
        rows.push(CompareRow {
            timestamp,
            run_index,
            fixture,
            threads,
            rvp9_p50_us,
        });
    }
    Ok(rows)
}

/// Median of an arbitrary slice. Returns `f64::NAN` for empty input.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[derive(Debug, Clone, Copy)]
enum Gate {
    Win,
    Regression,
    Noise,
}

impl Gate {
    fn classify(delta_percent: f64, p_value: f64) -> Gate {
        const ALPHA: f64 = 0.05;
        const DELTA_THRESHOLD: f64 = 5.0;
        if p_value < ALPHA {
            if delta_percent <= -DELTA_THRESHOLD {
                Gate::Win
            } else if delta_percent >= DELTA_THRESHOLD {
                Gate::Regression
            } else {
                Gate::Noise
            }
        } else {
            Gate::Noise
        }
    }

    fn render(&self) -> &'static str {
        match self {
            Gate::Win => "**WIN**",
            Gate::Regression => "**REGRESSION**",
            Gate::Noise => "NOISE",
        }
    }
}

fn main() {
    let args = parse_args();

    let rows = match parse_csv(&args.csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    if rows.is_empty() {
        eprintln!(
            "error: no valid schema-v3 rows found in {}",
            args.csv_path.display(),
        );
        std::process::exit(1);
    }

    // Bucket by (fixture, threads), separated into baseline/candidate
    // vectors of p50 samples. BTreeMap so output is deterministic.
    let mut groups: BTreeMap<(String, u32), (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut baseline_runs: u32 = 0;
    let mut candidate_runs: u32 = 0;
    for row in &rows {
        if row.timestamp == args.baseline_ts {
            baseline_runs = baseline_runs.max(row.run_index);
            groups
                .entry((row.fixture.clone(), row.threads))
                .or_default()
                .0
                .push(row.rvp9_p50_us);
        } else if row.timestamp == args.candidate_ts {
            candidate_runs = candidate_runs.max(row.run_index);
            groups
                .entry((row.fixture.clone(), row.threads))
                .or_default()
                .1
                .push(row.rvp9_p50_us);
        }
    }
    if groups.is_empty() {
        eprintln!(
            "error: no rows matched baseline {} or candidate {}",
            args.baseline_ts, args.candidate_ts
        );
        std::process::exit(1);
    }
    if let Some(((fixture, threads), (base, candidate))) = groups
        .iter()
        .find(|(_, (base, candidate))| base.is_empty() || candidate.is_empty())
    {
        eprintln!(
            "error: comparison group {fixture} at {threads} thread(s) is one-sided \
             (baseline samples={}, candidate samples={})",
            base.len(),
            candidate.len()
        );
        std::process::exit(1);
    }

    let mut md = String::new();
    md.push_str(&format!(
        "# Bench comparison: baseline {} vs candidate {}\n\n",
        args.baseline_ts, args.candidate_ts,
    ));
    md.push_str(&format!(
        "baseline runs observed: {baseline_runs}; candidate runs observed: {candidate_runs}\n\n",
    ));
    md.push_str("gate: **WIN** if p<0.05 and Δ ≤ -5%; **REGRESSION** if p<0.05 and Δ ≥ +5%; NOISE otherwise.\n\n");
    md.push_str("| fixture | threads | n_base | n_cand | base p50 (ms) | cand p50 (ms) | Δ% | U | p | gate |\n");
    md.push_str("|---|---|---|---|---|---|---|---|---|---|\n");

    let mut any_win = false;
    let mut any_regression = false;
    for ((fixture, threads), (base, cand)) in &groups {
        let base_med = median(base);
        let cand_med = median(cand);
        let delta_pct = if base_med > 0.0 {
            (cand_med - base_med) / base_med * 100.0
        } else {
            f64::NAN
        };
        let mwu = mann_whitney_u(base, cand);
        let gate = Gate::classify(delta_pct, mwu.p_value);
        match gate {
            Gate::Win => any_win = true,
            Gate::Regression => any_regression = true,
            Gate::Noise => {}
        }
        // Convert microseconds to milliseconds for readability.
        md.push_str(&format!(
            "| {fixture} | {threads} | {nb} | {nc} | {bms:.2} | {cms:.2} | {d:+.1}% | {u} | {p:.4} | {gate} |\n",
            nb = base.len(),
            nc = cand.len(),
            bms = base_med / 1000.0,
            cms = cand_med / 1000.0,
            d = delta_pct,
            u = mwu.u_statistic,
            p = mwu.p_value,
            gate = gate.render(),
        ));
    }

    println!("{md}");

    let out_path = args.out_path.unwrap_or_else(|| {
        workspace_root().join("bench").join("results").join(format!(
            "compare-{}-vs-{}.md",
            args.baseline_ts, args.candidate_ts,
        ))
    });
    if let Some(parent) = out_path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!("error: create {}: {e}", parent.display());
        std::process::exit(1);
    }
    if let Err(e) = std::fs::write(&out_path, &md) {
        eprintln!("error: write {}: {e}", out_path.display());
        std::process::exit(1);
    }
    eprintln!("wrote {}", out_path.display());

    if any_regression {
        // Non-zero exit so callers can wire this into CI: any
        // significant regression fails the comparison.
        std::process::exit(1);
    }
    let _ = any_win; // currently informational
}
