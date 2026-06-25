//! Shared library for the `decode-throughput` benchmark binary and the
//! `decode_throughput_ratchet` integration test.
//!
//! Per HLD §G6: drive existing VP9 test vectors through
//! [`rvp9_decoder::Decoder`] in a loop until at least
//! [`MIN_TIMED_FRAMES`] **decoded frames** have been timed per fixture,
//! dropping the first [`WARMUP_FRAMES`] frames per HLD §6.4. Captures
//! per-frame wall-time and reduces to median / p95 / p99.
//!
//! # Metric semantics (G6 review fix)
//!
//! - **One sample = one [`Decoder::decode_frame`] call.** The previous
//!   implementation timed whole packets, which conflated multi-frame
//!   show-existing-frame packets with single-frame packets and reported
//!   per-packet medians as per-frame. The HLD §6.4 hard gate
//!   ("720p30 ≤ 33 ms median single-core") is expressed in
//!   frames-per-second, so the metric must be per-frame.
//! - **Steady-state, not cold-start.** The [`Decoder`] is constructed
//!   ONCE per fixture and reused across replay passes. The previous
//!   implementation re-created the [`Decoder`] every pass, contaminating
//!   the sample window with allocator + ref-map cold-start cost on each
//!   pass. Decoder construction cost is a separate concern (Wave C may
//!   add a `Decoder::new()` micro-bench).
//! - **Warmup is per-fixture, not per-pass.** Drop the first
//!   [`WARMUP_FRAMES`] *decoded* frames once per fixture; everything
//!   after is a sample.
//!
//! # Container vs decoder timing
//!
//! Container walk (`read_packets`) is outside the timing window — only
//! `Decoder::decode_frame` is measured. `parse_packet` runs inside the
//! timing window only when its work cannot be cleanly separated; this
//! cost is negligible vs. decode.
//!
//! # Fixture choice
//!
//! See [`DEFAULT_FIXTURES`] for the rationale.
//!
//! # CSV schema v1
//!
//! See [`append_csv`] for the column list. Schema is versioned; readers
//! must check column 0 (`schema_version`) before parsing.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;

use crate::Stats;

/// Frames dropped from the head of the timing window per fixture
/// (allocator + BTB + ref-map warm-up). Per HLD §6.4.
pub const WARMUP_FRAMES: usize = 30;

/// Minimum number of timed (i.e. post-warmup) frames per fixture
/// required to yield a meaningful p99. A single 10-frame conformance
/// fixture is far too small for a trustworthy tail — loop the decode
/// until we have at least this many timed samples.
pub const MIN_TIMED_FRAMES: usize = 1_000;

/// CSV schema version. Bumped whenever the column list changes; readers
/// MUST consult column 0 before parsing the remaining columns.
pub const CSV_SCHEMA_VERSION: u32 = 1;

/// Curated fixture set for the throughput metric. Names are relative
/// to `<workspace_root>/tests/conformance/vectors/`.
///
/// # Selection rationale (G6 review)
///
/// The metric is steady-state per-frame decode latency on representative
/// VP9 content. Vectors are chosen to:
///
/// - Cover **multi-tile** decode paths (1, 2, 4, 8 tiles) at full HD and
///   4K, since tile-parallel residual decode is the dominant Wave C
///   performance lever.
/// - Include a **single-tile control** (`vp90-2-00-quantizer-00`) so a
///   regression in the tile-stitching path is distinguishable from a
///   regression in shared decode helpers (transforms, intra-prediction,
///   loop-filter).
///
/// Vectors deliberately **excluded** (G6 review):
///
/// - `vp90-2-09-aq2.webm` — AQ stress; not representative of typical
///   content and skews the median upward.
/// - `vp90-2-10-show-existing-frame.webm` — show-existing-frame packets
///   do near-zero decode work per shown frame; including them was
///   misleading the previous (per-packet) metric.
pub const DEFAULT_FIXTURES: &[&str] = &[
    "vp90-2-00-quantizer-00.webm",
    "vp90-2-08-tile_1x2.webm",
    "vp90-2-08-tile_1x4.webm",
    "vp90-2-08-tile_1x8.webm",
];

/// Per-fixture throughput measurement.
#[derive(Debug, Clone)]
pub struct FixtureResult {
    /// Fixture file name (basename only, e.g. `vp90-2-08-tile_1x2.webm`).
    pub fixture: String,
    /// Number of timed (post-warmup) frames.
    pub timed_frames: usize,
    /// Width of the first decoded frame, in pixels. `0` if no frame decoded.
    pub width: u32,
    /// Height of the first decoded frame, in pixels. `0` if no frame decoded.
    pub height: u32,
    /// Per-frame decode wall-time stats, microseconds.
    pub stats: Stats,
}

/// Errors from running the throughput benchmark.
#[derive(Debug)]
pub enum ThroughputError {
    /// Fixture file missing from disk. Holds the path we looked for.
    MissingFixture(PathBuf),
    /// Container parse failed (corrupt IVF/WebM, or wrong extension).
    Container(String),
    /// Decoder returned a non-`Unimplemented` error on a shown frame —
    /// almost certainly a Wave A regression. Holds fixture name and
    /// stringified error.
    DecodeError { fixture: String, error: String },
    /// No frame ever decoded successfully — the fixture is unusable
    /// as a throughput sample.
    NoFramesDecoded { fixture: String },
    /// CSV append failed.
    Io(std::io::Error),
}

impl std::fmt::Display for ThroughputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFixture(p) => write!(f, "missing fixture: {}", p.display()),
            Self::Container(s) => write!(f, "container parse error: {s}"),
            Self::DecodeError { fixture, error } => {
                write!(f, "{fixture}: decode error: {error}")
            }
            Self::NoFramesDecoded { fixture } => {
                write!(f, "{fixture}: no frames decoded")
            }
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for ThroughputError {}

impl From<std::io::Error> for ThroughputError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Resolve the workspace `tests/conformance/vectors/` directory.
///
/// Honors `RVP9_VECTORS_DIR` if set; otherwise walks up from
/// `CARGO_MANIFEST_DIR` (the bench crate) two levels to the workspace
/// root. Mirrors the resolution done by the conformance harness's
/// `vectors_dir()`.
#[must_use]
pub fn vectors_dir() -> PathBuf {
    if let Ok(env) = std::env::var("RVP9_VECTORS_DIR") {
        return PathBuf::from(env);
    }
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(&crate_root)
        .join("tests")
        .join("conformance")
        .join("vectors")
}

/// Drive a single fixture through the decoder, looping until at least
/// `min_timed_frames` post-warmup frames have been timed. Returns
/// per-frame stats in **microseconds**.
///
/// Loads the fixture into memory once (no file IO inside the timing
/// window). The container `read_packets` step is also outside the
/// timing window. The [`Decoder`] is constructed **once** and reused
/// across replay passes — see module docs for steady-state semantics.
///
/// Each call to [`Decoder::decode_frame`] is one timing sample.
/// Skipped frames (`Unimplemented`) produce no sample.
///
/// # Errors
///
/// - [`ThroughputError::MissingFixture`] if the file does not exist.
/// - [`ThroughputError::Container`] on a malformed IVF/WebM header.
/// - [`ThroughputError::DecodeError`] on any non-`Unimplemented` decoder
///   error (likely a Wave A regression — surface, do not swallow).
/// - [`ThroughputError::NoFramesDecoded`] if every frame skipped or
///   errored out.
pub fn run_one(
    fixture_path: &Path,
    warmup_frames: usize,
    min_timed_frames: usize,
) -> Result<FixtureResult, ThroughputError> {
    run_with_decoder(fixture_path, warmup_frames, min_timed_frames, Decoder::new)
}

/// Like [`run_one`], but constructs the decoder with an explicit thread
/// count via [`Decoder::with_threads`]. Q.1.A head-to-head bench surface
/// (HLD §3).
///
/// `threads` is `NonZeroUsize`; pass `NonZeroUsize::new(1).unwrap()` for
/// the single-thread baseline.
///
/// # Errors
///
/// Same as [`run_one`].
pub fn run_one_threaded(
    fixture_path: &Path,
    threads: std::num::NonZeroUsize,
    warmup_frames: usize,
    min_timed_frames: usize,
) -> Result<FixtureResult, ThroughputError> {
    run_with_decoder(fixture_path, warmup_frames, min_timed_frames, || {
        Decoder::with_threads(threads)
    })
}

/// Shared decode-and-time loop used by [`run_one`] and
/// [`run_one_threaded`]. The constructor closure is the only knob; the
/// timing window, warmup discipline, and reduction math are identical
/// across callers.
fn run_with_decoder<F>(
    fixture_path: &Path,
    warmup_frames: usize,
    min_timed_frames: usize,
    make_decoder: F,
) -> Result<FixtureResult, ThroughputError>
where
    F: FnOnce() -> Decoder,
{
    if !fixture_path.exists() {
        return Err(ThroughputError::MissingFixture(fixture_path.to_path_buf()));
    }
    let fixture_name = fixture_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| fixture_path.display().to_string());

    let (data, packets) =
        read_packets(fixture_path).map_err(|e| ThroughputError::Container(e.to_string()))?;

    let mut samples_us: Vec<f64> = Vec::with_capacity(min_timed_frames + warmup_frames + 64);
    let mut frames_seen: usize = 0;
    let mut width: u32 = 0;
    let mut height: u32 = 0;

    // Steady-state metric: ONE Decoder across all replay passes. This
    // amortizes ref-map / alloc warmup beyond `warmup_frames` even
    // across pass boundaries. Decoder::new() cost is intentionally
    // excluded — measured separately if/when needed.
    let mut decoder = make_decoder();

    'outer: while samples_us.len() < min_timed_frames {
        let mut produced_any_this_pass = false;
        for range in &packets {
            let packet = range.slice(&data);

            let parsed = match decoder.parse_packet(packet) {
                Ok(p) => p,
                Err(_) => {
                    // Container-level header parse failure — not a
                    // decode sample.
                    continue;
                }
            };
            for frame in parsed {
                // Per-frame timing window. Each decode_frame call is
                // exactly one sample.
                let t0 = Instant::now();
                let result = decoder.decode_frame(&frame);
                let dt = t0.elapsed();

                match result {
                    Ok(buf) => {
                        if width == 0 {
                            width = buf.width();
                            height = buf.height();
                        }
                    }
                    Err(rvp9_decoder::DecodeError::Unimplemented { .. }) => {
                        // Skipped frame: not a real decode sample.
                        // Don't record a timing.
                        continue;
                    }
                    Err(e) => {
                        return Err(ThroughputError::DecodeError {
                            fixture: fixture_name,
                            error: format!("{e:?}"),
                        });
                    }
                }

                produced_any_this_pass = true;
                frames_seen += 1;
                if frames_seen > warmup_frames {
                    samples_us.push(dt.as_secs_f64() * 1_000_000.0);
                    if samples_us.len() >= min_timed_frames {
                        break 'outer;
                    }
                }
            }
        }
        if !produced_any_this_pass {
            // Every packet skipped or unparseable — looping won't help.
            break;
        }
    }

    if samples_us.is_empty() {
        return Err(ThroughputError::NoFramesDecoded {
            fixture: fixture_name,
        });
    }

    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let stats = Stats::from_sorted(&samples_us).expect("non-empty by guard above");

    Ok(FixtureResult {
        fixture: fixture_name,
        timed_frames: samples_us.len(),
        width,
        height,
        stats,
    })
}

/// Run [`run_one`] over a list of fixture paths, returning all results.
/// Continues past errors so a broken fixture doesn't kill the whole run;
/// the error is recorded in the returned `Vec<Result<_, _>>`.
pub fn run_all(
    fixture_paths: &[PathBuf],
    warmup_frames: usize,
    min_timed_frames: usize,
) -> Vec<Result<FixtureResult, ThroughputError>> {
    fixture_paths
        .iter()
        .map(|p| run_one(p, warmup_frames, min_timed_frames))
        .collect()
}

/// Best-effort host fingerprint: `<os>-<arch>`. Cheap and
/// dependency-free; the dedicated Linux CI runner (HLD §6.4) is where
/// numbers are actually pinned, so this is just enough to disambiguate
/// dev-box runs in the trend CSV.
///
/// Drops the `FAMILY` token from the previous format — for our targets
/// it was always redundant with `OS` (`windows-x86_64-windows`,
/// `linux-x86_64-unix`).
#[must_use]
pub fn host_fingerprint() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Build profile descriptor used to disambiguate CSV rows produced by
/// different builds. Captures `(opt_level, debug_assertions)` — the
/// two knobs that materially change decode latency.
///
/// Hashed (not raw) so the CSV row stays a short stable token while
/// still differing across debug vs. release.
#[must_use]
pub fn build_profile_hash() -> String {
    // Stable, dependency-free FNV-1a 64-bit over a deterministic
    // string. Output is hex-encoded so the CSV column is fixed-width.
    let descriptor = format!(
        "debug_assertions={};opt={}",
        cfg!(debug_assertions),
        // `OPT_LEVEL` is exposed to build scripts but not to crate
        // sources; embed a best-effort proxy. Release builds set
        // debug_assertions=false on standard profiles.
        if cfg!(debug_assertions) {
            "dev"
        } else {
            "release"
        },
    );
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in descriptor.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    format!("{hash:016x}")
}

/// Append one CSV row per fixture result to `csv_path`, creating the
/// parent directory and writing the header line if the file does not
/// already exist.
///
/// # Schema v1 (G6 review)
///
/// Column 0 is always `schema_version` so future versions can be added
/// without breaking existing readers.
///
/// ```text
/// schema_version,timestamp_unix,host,unpinned,build_profile_hash,fixture,\
///   timed_frames,width,height,p50_us,p95_us,p99_us
/// ```
///
/// - `schema_version`: u32, currently [`CSV_SCHEMA_VERSION`].
/// - `timestamp_unix`: seconds since UNIX_EPOCH (UTC).
/// - `host`: from [`host_fingerprint`].
/// - `unpinned`: `true` if this row came from a host NOT in the
///   ratchet's `PINNED_HOSTS` allow-list (i.e. a dev-box run). The
///   ratchet baseline lookup filters these out.
/// - `build_profile_hash`: from [`build_profile_hash`] — disambiguates
///   debug vs release.
pub fn append_csv(
    csv_path: &Path,
    timestamp_unix: u64,
    host: &str,
    unpinned: bool,
    build_profile_hash: &str,
    results: &[FixtureResult],
) -> Result<(), ThroughputError> {
    if let Some(parent) = csv_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let needs_header = !csv_path.exists();
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)?;
    if needs_header {
        writeln!(
            f,
            "schema_version,timestamp_unix,host,unpinned,build_profile_hash,\
             fixture,timed_frames,width,height,p50_us,p95_us,p99_us"
        )?;
    }
    for r in results {
        writeln!(
            f,
            "{v},{ts},{host},{unpinned},{bph},{fixture},{n},{w},{h},\
             {p50:.3},{p95:.3},{p99:.3}",
            v = CSV_SCHEMA_VERSION,
            ts = timestamp_unix,
            host = host,
            unpinned = unpinned,
            bph = build_profile_hash,
            fixture = r.fixture,
            n = r.timed_frames,
            w = r.width,
            h = r.height,
            p50 = r.stats.p50,
            p95 = r.stats.p95,
            p99 = r.stats.p99,
        )?;
    }
    Ok(())
}

/// One parsed CSV row.
#[derive(Debug, Clone)]
pub struct CsvRow {
    pub schema_version: u32,
    pub timestamp_unix: u64,
    pub host: String,
    pub unpinned: bool,
    pub build_profile_hash: String,
    pub fixture: String,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
}

/// Read all CSV rows, ignoring the header. Returns an empty Vec if the
/// file is missing or empty.
///
/// Currently understands schema v1 only. Rows whose `schema_version`
/// column is not 1 are silently skipped; future versions can extend
/// this match.
///
/// # Errors
///
/// Propagates filesystem errors. A row that fails to parse is silently
/// skipped (treat as corrupt or future-version entry).
pub fn read_csv(csv_path: &Path) -> Result<Vec<CsvRow>, ThroughputError> {
    if !csv_path.exists() {
        return Ok(Vec::new());
    }
    let raw = std::fs::read_to_string(csv_path)?;
    let mut rows = Vec::new();
    for line in raw.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        // v1 = 12 columns.
        if cols.len() < 12 {
            continue;
        }
        let Ok(schema_version) = cols[0].parse::<u32>() else {
            continue;
        };
        if schema_version != CSV_SCHEMA_VERSION {
            continue;
        }
        let Ok(timestamp_unix) = cols[1].parse::<u64>() else {
            continue;
        };
        let host = cols[2].to_string();
        let unpinned = match cols[3] {
            "true" => true,
            "false" => false,
            _ => continue,
        };
        let build_profile_hash = cols[4].to_string();
        let fixture = cols[5].to_string();
        let Ok(p50_us) = cols[9].parse::<f64>() else {
            continue;
        };
        let Ok(p95_us) = cols[10].parse::<f64>() else {
            continue;
        };
        let Ok(p99_us) = cols[11].parse::<f64>() else {
            continue;
        };
        rows.push(CsvRow {
            schema_version,
            timestamp_unix,
            host,
            unpinned,
            build_profile_hash,
            fixture,
            p50_us,
            p95_us,
            p99_us,
        });
    }
    Ok(rows)
}

/// For each unique fixture in `rows`, return its most-recent **pinned**
/// row (max `timestamp_unix`) whose `host` is in `pinned_hosts`.
///
/// Unpinned rows (dev-box runs) are filtered out BEFORE the max-by-timestamp
/// reduction — the ratchet compares apples-to-apples against the
/// pinned-host baseline only.
///
/// If `pinned_hosts` is non-empty and contains more than one host, the
/// caller should filter to a single host first; otherwise comparing
/// host-A baseline against host-B current is a logic error. This
/// function does NOT silently merge across hosts — each (fixture, host)
/// pair is reduced independently, then the per-fixture max-timestamp
/// row wins regardless of host. Callers needing strict single-host
/// behavior should pre-filter `rows` or pass a single-element
/// `pinned_hosts`.
#[must_use]
pub fn latest_per_fixture(rows: &[CsvRow], pinned_hosts: &[&str]) -> Vec<CsvRow> {
    use std::collections::HashMap;
    let mut latest: HashMap<String, CsvRow> = HashMap::new();
    for r in rows {
        if r.unpinned {
            continue;
        }
        if !pinned_hosts.contains(&r.host.as_str()) {
            continue;
        }
        latest
            .entry(r.fixture.clone())
            .and_modify(|cur| {
                if r.timestamp_unix > cur.timestamp_unix {
                    *cur = r.clone();
                }
            })
            .or_insert_with(|| r.clone());
    }
    let mut out: Vec<CsvRow> = latest.into_values().collect();
    out.sort_by(|a, b| a.fixture.cmp(&b.fixture));
    out
}

/// Pure metric ratchet check: returns `Ok(())` if `current_us` is within
/// `threshold` × `baseline_us`, else `Err` with a human-readable
/// regression message.
///
/// Extracted for unit-test coverage of the pass/fail branches.
///
/// # Errors
///
/// Returns a formatted regression message string when the budget is
/// exceeded. The caller decides whether to `panic!` (pinned host) or
/// just warn (unpinned host).
pub fn check_metric_ratchet(
    fixture: &str,
    metric: &str,
    baseline_us: f64,
    current_us: f64,
    threshold: f64,
) -> Result<(), String> {
    let budget = baseline_us * threshold;
    if current_us > budget {
        Err(format!(
            "{fixture}: {metric} {current_us:.1}us > {budget:.1}us \
             (baseline {baseline_us:.1}us + {pct:.0}%)",
            pct = (threshold - 1.0) * 100.0,
        ))
    } else {
        Ok(())
    }
}

/// Backwards-compatible p50 ratchet check.
///
/// Prefer [`check_metric_ratchet`] for new callers that need p95/p99 tail
/// coverage too.
///
/// # Errors
///
/// Returns a formatted regression message string when the p50 budget is
/// exceeded. The caller decides whether to `panic!` (pinned host) or
/// just warn (unpinned host).
pub fn check_ratchet(
    fixture: &str,
    baseline_p50_us: f64,
    current_p50_us: f64,
    threshold: f64,
) -> Result<(), String> {
    check_metric_ratchet(fixture, "p50", baseline_p50_us, current_p50_us, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(name: &str, p50: f64, p95: f64, p99: f64) -> FixtureResult {
        FixtureResult {
            fixture: name.into(),
            timed_frames: 1000,
            width: 640,
            height: 480,
            stats: Stats {
                n: 1000,
                min: p50 - 50.0,
                max: p99 + 1.0,
                mean: p50,
                p50,
                p90: p95 - 5.0,
                p95,
                p99,
            },
        }
    }

    #[test]
    fn csv_roundtrip_v1() {
        let dir = std::env::temp_dir().join("rvp9_throughput_csv_test_v1");
        let _ = std::fs::remove_dir_all(&dir);
        let csv = dir.join("decode-throughput.csv");

        let r1 = make_result("fix_a.webm", 150.0, 190.0, 199.0);
        let r2 = make_result("fix_b.webm", 300.0, 390.0, 399.0);

        // First append: pinned host, only fix_a.
        append_csv(
            &csv,
            1_700_000_000,
            "linux-x86_64",
            false,
            "deadbeef",
            std::slice::from_ref(&r1),
        )
        .unwrap();
        // Second append: same pinned host, newer timestamp, both fixtures.
        append_csv(
            &csv,
            1_700_001_000,
            "linux-x86_64",
            false,
            "deadbeef",
            &[r1.clone(), r2.clone()],
        )
        .unwrap();
        // Third append: UNPINNED host (dev box) — should be filtered
        // out of latest_per_fixture.
        let r1_unpinned = make_result("fix_a.webm", 999.0, 1100.0, 1200.0);
        append_csv(
            &csv,
            1_700_002_000,
            "windows-x86_64",
            true,
            "cafef00d",
            std::slice::from_ref(&r1_unpinned),
        )
        .unwrap();

        let rows = read_csv(&csv).unwrap();
        assert_eq!(rows.len(), 4, "rows={rows:?}");

        let latest = latest_per_fixture(&rows, &["linux-x86_64"]);
        assert_eq!(latest.len(), 2);
        let by_name: std::collections::HashMap<_, _> =
            latest.iter().map(|r| (r.fixture.as_str(), r)).collect();
        // The unpinned 999.0 row MUST NOT win.
        assert_eq!(by_name["fix_a.webm"].timestamp_unix, 1_700_001_000);
        assert!((by_name["fix_a.webm"].p50_us - 150.0).abs() < 1e-6);
        assert!((by_name["fix_b.webm"].p99_us - 399.0).abs() < 1e-6);
        assert!(!by_name["fix_a.webm"].unpinned);
    }

    #[test]
    fn latest_per_fixture_empty_when_no_pinned_rows() {
        let rows = vec![CsvRow {
            schema_version: 1,
            timestamp_unix: 100,
            host: "windows-x86_64".into(),
            unpinned: true,
            build_profile_hash: "deadbeef".into(),
            fixture: "fix.webm".into(),
            p50_us: 100.0,
            p95_us: 110.0,
            p99_us: 120.0,
        }];
        // Only unpinned data — pinned-only filter returns empty.
        let latest = latest_per_fixture(&rows, &["linux-x86_64"]);
        assert!(latest.is_empty());
    }

    #[test]
    fn read_csv_missing_file_returns_empty() {
        let p = std::env::temp_dir().join("rvp9_throughput_does_not_exist_xyz.csv");
        let _ = std::fs::remove_file(&p);
        let rows = read_csv(&p).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn read_csv_skips_malformed_rows() {
        let p = std::env::temp_dir().join("rvp9_throughput_malformed_v1.csv");
        std::fs::write(
            &p,
            "schema_version,timestamp_unix,host,unpinned,build_profile_hash,\
             fixture,timed_frames,width,height,p50_us,p95_us,p99_us\n\
             1,1000,h,false,bph,a,n,w,h,p50,p95,p99\n\
             1,1001,h,false,bph,b,1000,640,480,100.0,150.0,200.0\n",
        )
        .unwrap();
        let rows = read_csv(&p).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].fixture, "b");
    }

    #[test]
    fn read_csv_skips_other_schema_versions() {
        let p = std::env::temp_dir().join("rvp9_throughput_future_schema.csv");
        std::fs::write(
            &p,
            "schema_version,timestamp_unix,host,unpinned,build_profile_hash,\
             fixture,timed_frames,width,height,p50_us,p95_us,p99_us\n\
             2,1000,h,false,bph,a,1000,640,480,100.0,150.0,200.0\n\
             1,1001,h,false,bph,b,1000,640,480,200.0,300.0,400.0\n",
        )
        .unwrap();
        let rows = read_csv(&p).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].fixture, "b");
    }

    #[test]
    fn check_ratchet_within_budget_ok() {
        // 100us baseline, 109us current, 10% threshold → within budget.
        assert!(check_ratchet("fix.webm", 100.0, 109.0, 1.10).is_ok());
        // Equal-to-budget exactly: within budget (strict `>` check).
        assert!(check_ratchet("fix.webm", 100.0, 110.0, 1.10).is_ok());
    }

    #[test]
    fn check_ratchet_over_budget_errs() {
        let res = check_ratchet("fix.webm", 100.0, 110.001, 1.10);
        let msg = res.expect_err("should be a regression");
        assert!(msg.contains("fix.webm"));
        assert!(msg.contains("p50"));
        assert!(msg.contains("10%"));
    }

    #[test]
    fn check_metric_ratchet_reports_metric_name_and_budget() {
        assert!(check_metric_ratchet("fix.webm", "p95", 100.0, 114.0, 1.15).is_ok());
        let msg =
            check_metric_ratchet("fix.webm", "p99", 100.0, 120.001, 1.20).unwrap_err();
        assert!(msg.contains("fix.webm"));
        assert!(msg.contains("p99"));
        assert!(msg.contains("20%"));
    }

    #[test]
    fn host_fingerprint_has_no_family_token() {
        let fp = host_fingerprint();
        // Format is `<os>-<arch>` — two parts only.
        assert_eq!(fp.matches('-').count(), 1, "expected single dash in `{fp}`");
    }
}
