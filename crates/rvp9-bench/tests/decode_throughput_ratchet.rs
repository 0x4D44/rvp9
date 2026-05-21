//! Soft-ratchet for the `decode-throughput` benchmark.
//!
//! Per HLD §G6: re-runs the throughput library function in-process and
//! asserts current p50 (median) decode time stays within a 10% regression
//! budget relative to the last committed **pinned-host** baseline row
//! per fixture in `bench/results/decode-throughput.csv`.
//!
//! On a fresh repo (no pinned baseline yet), the test PASSES and prints
//! "no pinned baseline yet — pinned hosts only" so the test is
//! self-bootstrapping: dev-box runs can still append CSV rows (with
//! `unpinned=true`), but only rows from a host in [`PINNED_HOSTS`] are
//! used as the ratchet baseline.
//!
//! This is a SOFT gate (10% regression budget on p50 only). The hard
//! real-time gate (720p30 ≤ 33ms median single-core etc.) lands as
//! part of Wave C exit per HLD §6.4.
//!
//! Per HLD §6.4: Windows dev-box runs are informational only (the
//! dedicated Linux CI runner is the gate host). When the test is run on
//! a host not on the [`PINNED_HOSTS`] allow-list, ratchet violations
//! are downgraded to warnings rather than test failures — otherwise
//! every developer running `cargo test` locally would get spurious
//! noise failures from missing `taskset`/`cpupower` discipline.

use std::path::{Path, PathBuf};

use rvp9_bench::throughput::{
    self, FixtureResult, MIN_TIMED_FRAMES, WARMUP_FRAMES, check_ratchet, host_fingerprint,
    latest_per_fixture, read_csv,
};

/// 10% regression budget per HLD §G6.
const RATCHET_RATIO: f64 = 1.10;

/// Host fingerprints (from [`host_fingerprint`]) where the ratchet
/// is a HARD gate AND whose rows are used as the baseline.
///
/// Empty for now — the dedicated Linux CI runner (HLD §6.4) hasn't been
/// provisioned yet. Add its exact `<os>-<arch>` fingerprint
/// here once it's online so the ratchet starts gating on it.
///
/// Note: keep this list short. Cross-host comparison (e.g. Linux-runner-A
/// baseline vs Linux-runner-B current) is not meaningful — micro-arch
/// and cache hierarchy differences trivially exceed the 10% budget.
const PINNED_HOSTS: &[&str] = &[];

fn workspace_root() -> PathBuf {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or(crate_root)
}

fn baseline_csv() -> PathBuf {
    workspace_root()
        .join("bench")
        .join("results")
        .join("decode-throughput.csv")
}

fn fixtures_present(vectors_dir: &Path) -> Vec<PathBuf> {
    throughput::DEFAULT_FIXTURES
        .iter()
        .filter_map(|f| {
            let p = vectors_dir.join(f);
            if p.exists() { Some(p) } else { None }
        })
        .collect()
}

#[test]
fn decode_throughput_ratchet() {
    // The committed baseline is produced from `cargo run --release`;
    // running this test in a debug build would compare debug-mode
    // timings against a release-mode baseline and trip the ratchet
    // every time. Skip in debug mode and direct callers at the
    // canonical invocation.
    if cfg!(debug_assertions) {
        eprintln!(
            "[ratchet] skipping in debug build — baseline is release-mode. \
             Run `cargo test -p rvp9-bench --release \
             --test decode_throughput_ratchet` instead."
        );
        return;
    }

    let vectors = throughput::vectors_dir();
    let fixtures = fixtures_present(&vectors);

    if fixtures.is_empty() {
        eprintln!(
            "no conformance fixtures present at {} — \
             run scripts/fetch-vectors.sh to populate, then \
             rerun this test. Skipping ratchet check.",
            vectors.display(),
        );
        return;
    }

    // Use the SAME timed-frame floor as the bench binary so the
    // ratchet compares apples-to-apples against the committed baseline.
    // A smaller floor produces a noisier median that trips the 10%
    // ratchet against a 1000-frame baseline even on a clean run.
    let test_min_timed: usize = MIN_TIMED_FRAMES;

    let results: Vec<FixtureResult> = fixtures
        .iter()
        .filter_map(
            |p| match throughput::run_one(p, WARMUP_FRAMES, test_min_timed) {
                Ok(r) => Some(r),
                Err(e) => {
                    eprintln!("[ratchet] {}: {e}", p.display());
                    None
                }
            },
        )
        .collect();

    if results.is_empty() {
        eprintln!("[ratchet] no fixtures produced timed samples; skipping ratchet");
        return;
    }

    let csv_path = baseline_csv();
    let rows = read_csv(&csv_path).expect("baseline CSV read failed");
    if rows.is_empty() {
        eprintln!(
            "[ratchet] first run — no baseline at {}; \
             ratchet starts gating once a pinned-host row is committed.",
            csv_path.display(),
        );
        return;
    }

    let baseline = latest_per_fixture(&rows, PINNED_HOSTS);
    if baseline.is_empty() {
        eprintln!(
            "[ratchet] no pinned baseline yet — pinned hosts only. \
             ({} CSV row(s) on file, all unpinned or off-pinned-list.) \
             Ratchet starts gating once a row from a host in PINNED_HOSTS \
             is committed; see HLD §6.4.",
            rows.len(),
        );
        return;
    }
    let by_name: std::collections::HashMap<&str, &throughput::CsvRow> =
        baseline.iter().map(|r| (r.fixture.as_str(), r)).collect();

    let host = host_fingerprint();
    let pinned = PINNED_HOSTS.contains(&host.as_str());

    let mut regressions: Vec<String> = Vec::new();
    let mut compared: usize = 0;
    for r in &results {
        let Some(base) = by_name.get(r.fixture.as_str()) else {
            eprintln!(
                "[ratchet] {}: no baseline row yet — skipping (will be \
                 added on next bench-binary run + commit)",
                r.fixture,
            );
            continue;
        };
        compared += 1;
        match check_ratchet(&r.fixture, base.p50_us, r.stats.p50, RATCHET_RATIO) {
            Ok(()) => {
                let budget = base.p50_us * RATCHET_RATIO;
                eprintln!(
                    "[ratchet] {fix}: OK   p50 {cur:.1}us <= {bud:.1}us (baseline {bp:.1}us)",
                    fix = r.fixture,
                    cur = r.stats.p50,
                    bud = budget,
                    bp = base.p50_us,
                );
            }
            Err(msg) => {
                regressions.push(msg);
            }
        }
    }

    if !regressions.is_empty() {
        let msg = format!(
            "decode-throughput ratchet: {} of {compared} fixtures regressed \
             on host `{host}`:\n  {}",
            regressions.len(),
            regressions.join("\n  ")
        );
        if pinned {
            panic!("{msg}");
        } else {
            // Unpinned host (per HLD §6.4 — no taskset/cpupower
            // discipline): downgrade to warning. The HARD gate runs
            // only on the dedicated CI runner once it's online.
            eprintln!(
                "WARN: {msg}\n\
                 (host `{host}` not in PINNED_HOSTS — soft ratchet; \
                 see HLD §6.4)"
            );
        }
    }

    if compared == 0 {
        eprintln!(
            "[ratchet] no fixture had a baseline row; nothing compared. \
             Run the bench binary + commit `decode-throughput.csv` to \
             seed the baseline."
        );
    }
}
