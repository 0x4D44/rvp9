//! Soft-ratchet for the `decode-throughput` benchmark.
//!
//! Per HLD §G6: re-runs the throughput library function in-process and
//! asserts current p50/p95/p99 decode time stays within regression budgets
//! relative to the last committed **pinned-host, same-build-profile** baseline
//! row per fixture in `bench/results/decode-throughput.csv`.
//!
//! The baseline must match the current host and build profile exactly. On a
//! fresh repo (or an unseen host/profile), the test passes with a diagnostic
//! and no comparison. Dev-box rows (`unpinned=true`) are never eligible.
//!
//! This is a SOFT gate (10% p50, 15% p95, 20% p99 budgets). The hard
//! real-time gate (720p30 ≤ 33ms median single-core etc.) lands as
//! part of Wave C exit per HLD §6.4.
//!
//! Per HLD §6.4: Windows dev-box runs are informational only (the
//! dedicated Linux CI runner is the gate host). A host outside the
//! [`PINNED_HOSTS`] allow-list has no eligible pinned baseline. For an
//! eligible comparison, the existing hard and soft gate handling applies.

use std::path::{Path, PathBuf};

use rvp9_bench::throughput::{
    self, FixtureResult, MIN_TIMED_FRAMES, WARMUP_FRAMES, build_profile_hash, check_metric_ratchet,
    host_fingerprint, latest_per_fixture_for_build, read_csv,
};

/// Median regression budget per HLD §G6.
const P50_RATCHET_RATIO: f64 = 1.10;
/// Tail samples are noisier than the median, but still catch real stalls.
const P95_RATCHET_RATIO: f64 = 1.15;
/// p99 is the widest soft ratchet because it is sensitive to OS jitter.
const P99_RATCHET_RATIO: f64 = 1.20;

/// Host fingerprints (from [`host_fingerprint`]) where the ratchet
/// is a HARD gate and whose current host may use matching rows as its
/// baseline. Rows from other hosts or build profiles are never compared.
///
/// Empty for now — the dedicated Linux CI runner (HLD §6.4) hasn't been
/// provisioned yet. Add its exact `<os>-<arch>` fingerprint
/// here once it's online so the ratchet starts gating on it.
///
/// Note: keep this list short. Cross-host comparison (e.g. Linux-runner-A
/// baseline vs Linux-runner-B current) is not meaningful — micro-arch
/// and cache hierarchy differences trivially exceed the 10% budget.
const PINNED_HOSTS: &[&str] = &[];

fn pinned_baseline_for_build(
    rows: &[throughput::CsvRow],
    host: &str,
    build_profile_hash: &str,
    pinned_hosts: &[&str],
) -> Vec<throughput::CsvRow> {
    if !pinned_hosts.contains(&host) {
        return Vec::new();
    }
    latest_per_fixture_for_build(rows, host, build_profile_hash)
}

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
fn ratchet_baseline_requires_allowlisted_current_identity() {
    let current = throughput::CsvRow {
        schema_version: throughput::CSV_SCHEMA_VERSION,
        timestamp_unix: 10,
        host: "current-host".into(),
        unpinned: false,
        build_profile_hash: "current-profile".into(),
        fixture: "fixture.webm".into(),
        p50_us: 10.0,
        p95_us: 11.0,
        p99_us: 12.0,
    };
    let rows = vec![
        current.clone(),
        throughput::CsvRow {
            timestamp_unix: 20,
            host: "other-host".into(),
            ..current.clone()
        },
        throughput::CsvRow {
            timestamp_unix: 30,
            build_profile_hash: "other-profile".into(),
            ..current
        },
    ];

    let baseline = pinned_baseline_for_build(
        &rows,
        "current-host",
        "current-profile",
        &["current-host", "other-host"],
    );
    assert_eq!(baseline.len(), 1);
    assert_eq!(baseline[0].timestamp_unix, 10);

    assert!(
        pinned_baseline_for_build(&rows, "current-host", "current-profile", &["other-host"],)
            .is_empty(),
        "a host outside the allow-list has no eligible pinned baseline"
    );
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
    let host = host_fingerprint();
    let current_build_profile_hash = build_profile_hash();
    let pinned = PINNED_HOSTS.contains(&host.as_str());
    if rows.is_empty() {
        eprintln!(
            "[ratchet] first run — no baseline at {}; \
             ratchet starts gating once the current host `{host}` is in \
             PINNED_HOSTS and a pinned row for build profile \
             `{current_build_profile_hash}` is committed.",
            csv_path.display(),
        );
        return;
    }

    let baseline =
        pinned_baseline_for_build(&rows, &host, &current_build_profile_hash, PINNED_HOSTS);
    if baseline.is_empty() {
        eprintln!(
            "[ratchet] no eligible baseline for current host `{host}` and \
             build profile `{current_build_profile_hash}` ({} CSV row(s) on file). \
             The current host {allowlist_status} PINNED_HOSTS; eligible rows \
             must be pinned, valid, and match this exact host and profile. \
             Commit a matching baseline row; see HLD §6.4.",
            rows.len(),
            allowlist_status = if pinned { "is in" } else { "is not in" },
        );
        return;
    }
    let by_name: std::collections::HashMap<&str, &throughput::CsvRow> =
        baseline.iter().map(|r| (r.fixture.as_str(), r)).collect();

    let mut regressions: Vec<String> = Vec::new();
    let mut compared: usize = 0;
    for r in &results {
        let Some(base) = by_name.get(r.fixture.as_str()) else {
            eprintln!(
                "[ratchet] {}: no exact same-host/profile baseline row yet — \
                 skipping (seed it with a pinned run on this host and build profile)",
                r.fixture,
            );
            continue;
        };
        compared += 1;
        let metrics = [
            ("p50", base.p50_us, r.stats.p50, P50_RATCHET_RATIO),
            ("p95", base.p95_us, r.stats.p95, P95_RATCHET_RATIO),
            ("p99", base.p99_us, r.stats.p99, P99_RATCHET_RATIO),
        ];

        let mut fixture_regressed = false;
        for (metric, baseline, current, ratio) in metrics {
            if let Err(msg) = check_metric_ratchet(&r.fixture, metric, baseline, current, ratio) {
                fixture_regressed = true;
                regressions.push(msg);
            }
        }

        if !fixture_regressed {
            eprintln!(
                "[ratchet] {fix}: OK   p50 {p50:.1}us <= {p50b:.1}us; \
                 p95 {p95:.1}us <= {p95b:.1}us; p99 {p99:.1}us <= {p99b:.1}us",
                fix = r.fixture,
                p50 = r.stats.p50,
                p50b = base.p50_us * P50_RATCHET_RATIO,
                p95 = r.stats.p95,
                p95b = base.p95_us * P95_RATCHET_RATIO,
                p99 = r.stats.p99,
                p99b = base.p99_us * P99_RATCHET_RATIO,
            );
        }
    }

    if !regressions.is_empty() {
        let msg = format!(
            "decode-throughput ratchet: {} metric regression(s) across {compared} fixture(s) \
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
            "[ratchet] no fixture had an exact same-host/profile baseline row; \
             nothing compared. Run the bench binary on this host and build profile, \
             then commit `decode-throughput.csv` to seed the baseline."
        );
    }
}
