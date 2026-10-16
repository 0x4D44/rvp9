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
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
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
    /// A packet from an otherwise valid container is not valid VP9.
    PacketParse {
        /// Fixture basename.
        fixture: String,
        /// 1-based packet index.
        packet_index: usize,
        /// Stringified bitstream parse error.
        error: String,
    },
    /// Decoder returned a non-`Unimplemented` error on a shown frame —
    /// almost certainly a Wave A regression. Holds fixture name and
    /// stringified error.
    DecodeError { fixture: String, error: String },
    /// No frame ever decoded successfully — the fixture is unusable
    /// as a throughput sample.
    NoFramesDecoded { fixture: String },
    /// CSV read, validation or append failed.
    Io(std::io::Error),
}

impl std::fmt::Display for ThroughputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFixture(p) => write!(f, "missing fixture: {}", p.display()),
            Self::Container(s) => write!(f, "container parse error: {s}"),
            Self::PacketParse {
                fixture,
                packet_index,
                error,
            } => write!(
                f,
                "{fixture}: packet {packet_index}: VP9 parse error: {error}"
            ),
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
/// - [`ThroughputError::PacketParse`] on malformed VP9 packet data.
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
        for (packet_offset, range) in packets.iter().enumerate() {
            let packet = range.slice(&data);

            let parsed =
                decoder
                    .parse_packet(packet)
                    .map_err(|error| ThroughputError::PacketParse {
                        fixture: fixture_name.clone(),
                        packet_index: packet_offset + 1,
                        error: format!("{error:?}"),
                    })?;
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
/// already exist or is empty. All counts, dimensions and serialized metrics
/// must be positive; metrics must also be finite. The entire batch is validated
/// before any filesystem changes. Text fields use CSV quoting and doubled quotes.
/// Existing files require read access to inspect their final byte. A missing
/// final record separator is added before appending rows. Cooperating calls take
/// an exclusive lock on the CSV file while selecting the header and appending
/// the complete batch; the lock is released when the file handle is dropped.
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
    // Validate the whole batch before even creating the destination directory.
    for r in results {
        if r.timed_frames == 0
            || r.width == 0
            || r.height == 0
            || !valid_metrics(r.stats.p50, r.stats.p95, r.stats.p99)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "{}: CSV counts, dimensions and metrics must be positive and finite",
                    r.fixture
                ),
            )
            .into());
        }
    }
    if let Some(parent) = csv_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(csv_path)?;
    f.lock()?;
    if f.metadata()?.len() == 0 {
        writeln!(f, "{}", CSV_HEADER.join(","))?;
    } else if !results.is_empty() {
        f.seek(SeekFrom::End(-1))?;
        let mut last_byte = [0];
        f.read_exact(&mut last_byte)?;
        if last_byte[0] != b'\n' {
            writeln!(f)?;
        }
    }
    for r in results {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            CSV_SCHEMA_VERSION,
            timestamp_unix,
            csv_field(host),
            unpinned,
            csv_field(build_profile_hash),
            csv_field(&r.fixture),
            r.timed_frames,
            r.width,
            r.height,
            csv_metric(r.stats.p50),
            csv_metric(r.stats.p95),
            csv_metric(r.stats.p99),
        )?;
    }
    Ok(())
}

const CSV_HEADER: [&str; 12] = [
    "schema_version",
    "timestamp_unix",
    "host",
    "unpinned",
    "build_profile_hash",
    "fixture",
    "timed_frames",
    "width",
    "height",
    "p50_us",
    "p95_us",
    "p99_us",
];

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn valid_metrics(p50: f64, p95: f64, p99: f64) -> bool {
    [p50, p95, p99].into_iter().all(positive_finite)
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn csv_metric(value: f64) -> String {
    let rounded = format!("{value:.3}");
    if rounded == "0.000" {
        value.to_string()
    } else {
        rounded
    }
}

fn invalid_csv(message: &str) -> ThroughputError {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message).into()
}

/// Decode one logical record. Quoting follows RFC 4180; LF record endings
/// and UTF-8 text are accepted alongside CRLF. Invalid quoting fails closed
/// because an embedded newline makes recovery at the next physical line unsafe.
fn next_csv_record(
    input: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<Option<Vec<String>>, ThroughputError> {
    #[derive(Clone, Copy)]
    enum State {
        Start,
        Unquoted,
        Quoted,
        Closed,
    }
    use State::*;
    if input.peek().is_none() {
        return Ok(None);
    }
    let mut state = Start;
    let mut fields = Vec::new();
    let mut field = String::new();
    while let Some(ch) = input.next() {
        if matches!(state, Quoted) {
            if ch == '"' {
                if input.peek() == Some(&'"') {
                    input.next();
                    field.push('"');
                } else {
                    state = Closed;
                }
            } else {
                field.push(ch);
            }
            continue;
        }
        match ch {
            ',' => {
                fields.push(std::mem::take(&mut field));
                state = Start;
            }
            '\n' | '\r' => {
                if ch == '\r' && input.next() != Some('\n') {
                    return Err(invalid_csv("CSV record ending must be LF or CRLF"));
                }
                fields.push(field);
                return Ok(Some(fields));
            }
            '"' if matches!(state, Start) => state = Quoted,
            _ if matches!(state, Closed) => {
                return Err(invalid_csv("unexpected text after CSV closing quote"));
            }
            '"' => return Err(invalid_csv("quote in unquoted CSV field")),
            _ => {
                field.push(ch);
                state = Unquoted;
            }
        }
    }
    if matches!(state, Quoted) {
        return Err(invalid_csv("unterminated quoted CSV field"));
    }
    fields.push(field);
    Ok(Some(fields))
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

/// Read valid schema-v1 CSV rows. Missing or empty files yield no rows.
///
/// The first record must contain the twelve schema-v1 column names in order.
/// LF and CRLF record endings and quoted commas, quotes and newlines are accepted.
/// Well-formed records with another schema, the wrong column count, invalid
/// numeric/boolean fields, nonpositive counts/dimensions or nonfinite/nonpositive
/// metrics are skipped. Blank records are also skipped after the header.
///
/// # Errors
///
/// Propagates filesystem errors. A wrong header or malformed CSV quoting/record
/// ending returns [`std::io::ErrorKind::InvalidData`] for the whole file: the
/// next physical line cannot safely identify a new record after a quote error.
pub fn read_csv(csv_path: &Path) -> Result<Vec<CsvRow>, ThroughputError> {
    let raw = match std::fs::read_to_string(csv_path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut input = raw.chars().peekable();
    let Some(header) = next_csv_record(&mut input)? else {
        return Ok(Vec::new());
    };
    if header != CSV_HEADER {
        return Err(invalid_csv("unexpected throughput CSV header"));
    }
    let mut rows = Vec::new();
    while let Some(cols) = next_csv_record(&mut input)? {
        if let Some(row) = parse_csv_row(&cols) {
            rows.push(row);
        }
    }
    Ok(rows)
}

fn parse_csv_row(cols: &[String]) -> Option<CsvRow> {
    if cols.len() != CSV_HEADER.len() {
        return None;
    }
    let schema_version = cols[0].parse::<u32>().ok()?;
    if schema_version != CSV_SCHEMA_VERSION {
        return None;
    }
    let timestamp_unix = cols[1].parse::<u64>().ok()?;
    let unpinned = cols[3].parse::<bool>().ok()?;
    let timed_frames = cols[6].parse::<usize>().ok()?;
    let width = cols[7].parse::<u32>().ok()?;
    let height = cols[8].parse::<u32>().ok()?;
    let p50_us = cols[9].parse::<f64>().ok()?;
    let p95_us = cols[10].parse::<f64>().ok()?;
    let p99_us = cols[11].parse::<f64>().ok()?;
    if timed_frames == 0 || width == 0 || height == 0 || !valid_metrics(p50_us, p95_us, p99_us) {
        return None;
    }
    Some(CsvRow {
        schema_version,
        timestamp_unix,
        host: cols[2].clone(),
        unpinned,
        build_profile_hash: cols[4].clone(),
        fixture: cols[5].clone(),
        p50_us,
        p95_us,
        p99_us,
    })
}

/// For each unique fixture in `rows`, return its most-recent **pinned**
/// row (max `timestamp_unix`) whose `host` is in `pinned_hosts`.
/// Rows with nonfinite or nonpositive metrics are excluded, including rows
/// constructed directly through the public [`CsvRow`] fields.
///
/// Unpinned rows (dev-box runs) are filtered out BEFORE the max-by-timestamp
/// reduction — the ratchet compares apples-to-apples against the
/// pinned-host baseline only.
///
/// If `pinned_hosts` contains more than one host, the newest row for a
/// fixture may come from any of those hosts. This preserves the legacy
/// cross-host selection behavior; callers comparing a current run should
/// use [`latest_per_fixture_for_build`] instead.
#[must_use]
pub fn latest_per_fixture(rows: &[CsvRow], pinned_hosts: &[&str]) -> Vec<CsvRow> {
    latest_per_fixture_matching(rows, |r| {
        !r.unpinned
            && pinned_hosts.contains(&r.host.as_str())
            && valid_metrics(r.p50_us, r.p95_us, r.p99_us)
    })
}

/// For each fixture, return its newest valid pinned row for an exact host
/// and build profile.
///
/// Unpinned rows and rows with nonfinite or nonpositive metrics are
/// excluded before selecting the newest timestamp. The output is sorted
/// by fixture name. The caller remains responsible for deciding whether
/// `host` is in its pinned-host allow-list.
#[must_use]
pub fn latest_per_fixture_for_build(
    rows: &[CsvRow],
    host: &str,
    build_profile_hash: &str,
) -> Vec<CsvRow> {
    latest_per_fixture_matching(rows, |r| {
        !r.unpinned
            && r.host == host
            && r.build_profile_hash == build_profile_hash
            && valid_metrics(r.p50_us, r.p95_us, r.p99_us)
    })
}

fn latest_per_fixture_matching(
    rows: &[CsvRow],
    mut is_eligible: impl FnMut(&CsvRow) -> bool,
) -> Vec<CsvRow> {
    use std::collections::HashMap;
    let mut latest: HashMap<String, CsvRow> = HashMap::new();
    for r in rows {
        if !is_eligible(r) {
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
/// Returns an error if an operand or computed budget is nonfinite or
/// nonpositive, or a formatted regression message when the budget is
/// exceeded. The caller decides whether to `panic!` (pinned host) or
/// just warn (unpinned host).
pub fn check_metric_ratchet(
    fixture: &str,
    metric: &str,
    baseline_us: f64,
    current_us: f64,
    threshold: f64,
) -> Result<(), String> {
    if ![baseline_us, current_us, threshold]
        .into_iter()
        .all(positive_finite)
    {
        return Err(format!(
            "{fixture}: {metric} baseline, current and threshold must be finite and positive"
        ));
    }
    let budget = baseline_us * threshold;
    if !positive_finite(budget) {
        return Err(format!(
            "{fixture}: {metric} computed budget must be finite and positive"
        ));
    }
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
/// Returns an error if an input or computed budget is nonfinite or
/// nonpositive, or a formatted regression message when the p50 budget is
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
    use rvp9_harness::container::{IvfHeader, write_ivf};

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

    const TEST_HEADER: &str = "schema_version,timestamp_unix,host,unpinned,build_profile_hash,fixture,timed_frames,width,height,p50_us,p95_us,p99_us\n";

    fn csv_test_path(name: &str) -> PathBuf {
        let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch");
        std::fs::create_dir_all(&scratch).unwrap();
        scratch.join(format!("00108-{name}-{}.csv", std::process::id()))
    }

    fn read_literal(name: &str, body: &str) -> Result<Vec<CsvRow>, ThroughputError> {
        let path = csv_test_path(name);
        std::fs::write(&path, body).unwrap();
        let result = read_csv(&path);
        std::fs::remove_file(path).unwrap();
        result
    }

    #[test]
    fn csv_00108_reads_independent_quoted_multiline_oracle() {
        let literal = format!(
            "{TEST_HEADER}1,7,\"h,é\",false,\"b\"\"p\",\"a\r\nb\nc\rd,\"\"x\"\"\",1,2,3,1,2,3\r\n\n1,8,h,true,b,plain,4,5,6,4,5,6"
        );
        let rows = read_literal("quoted", &literal).unwrap();
        assert_eq!(rows.len(), 2, "quoted logical records must survive");
        assert_eq!(rows[0].host, "h,é");
        assert_eq!(rows[0].build_profile_hash, "b\"p");
        assert_eq!(rows[0].fixture, "a\r\nb\nc\rd,\"x\"");
        assert_eq!(rows[1].timestamp_unix, 8);
    }

    #[test]
    fn csv_00108_append_preserves_separatorless_existing_records() {
        let mut outcomes = Vec::new();
        for (name, suffix, expected_rows) in [
            ("header-eof", "", 1),
            ("data-eof", "\n1,6,h,false,b,old,1,2,3,1,2,3", 2),
        ] {
            let path = csv_test_path(name);
            std::fs::write(&path, format!("{}{suffix}", TEST_HEADER.trim_end())).unwrap();
            append_csv(
                &path,
                7,
                "h",
                false,
                "b",
                &[make_result("new", 1.0, 2.0, 3.0)],
            )
            .unwrap();
            let outcome = read_csv(&path).map(|rows| {
                rows.len() == expected_rows && rows.last().is_some_and(|row| row.fixture == "new")
            });
            std::fs::remove_file(path).unwrap();
            outcomes.push((name, outcome));
        }
        assert!(
            outcomes
                .iter()
                .all(|(_, outcome)| matches!(outcome, Ok(true))),
            "both separatorless header and data records must survive append: {outcomes:?}"
        );
    }

    #[test]
    fn csv_00108_writer_matches_independent_literal_oracle() {
        let path = csv_test_path("writer");
        std::fs::write(&path, "").unwrap();
        append_csv(
            &path,
            7,
            "h,é",
            false,
            "b\"p",
            &[make_result("a\r\nb\nc\rd,\"x\"", 1.0, 2.0, 3.0)],
        )
        .unwrap();
        let actual = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            actual,
            format!(
                "{TEST_HEADER}1,7,\"h,é\",false,\"b\"\"p\",\"a\r\nb\nc\rd,\"\"x\"\"\",1000,640,480,1.000,2.000,3.000\n"
            )
        );
    }

    #[test]
    fn csv_00108_skips_invalid_typed_rows() {
        let valid = [
            "1", "7", "h", "false", "b", "fixture", "1", "2", "3", "1", "2", "3",
        ];
        let mut body = TEST_HEADER.to_string();
        body.push_str(&format!("{},extra\n", valid.join(",")));
        body.push_str(&format!("{}\n", valid[..11].join(",")));
        for (index, values) in [
            (0, vec!["2", "x"]),
            (1, vec!["-1", "x"]),
            (3, vec!["TRUE"]),
            (6, vec!["x", "0", "-1", "18446744073709551616"]),
            (7, vec!["x", "0", "-1", "4294967296"]),
            (8, vec!["x", "0", "-1", "4294967296"]),
            (9, vec!["NaN", "inf", "-inf", "0", "-1"]),
            (10, vec!["NaN", "inf", "-inf", "0", "-1"]),
            (11, vec!["NaN", "inf", "-inf", "0", "-1"]),
        ] {
            for value in values {
                let mut cols = valid;
                cols[index] = value;
                body.push_str(&format!("{}\n", cols.join(",")));
            }
        }
        body.push_str(&valid.join(","));
        let rows = read_literal("typed", &body).unwrap();
        assert_eq!(
            rows.len(),
            1,
            "invalid schema/typed/metric/column rows must be skipped"
        );
    }

    #[test]
    fn csv_00108_rejects_ambiguous_quotes_and_bad_headers() {
        for suffix in [
            "1,7,h\"x,false,b,f,1,2,3,1,2,3\n",
            "1,7,\"h\"x,false,b,f,1,2,3,1,2,3\n",
            "1,7,\"unterminated",
            "1,7,h,false,b,f,1,2,3,1,2,3\r",
        ] {
            let body = format!("{TEST_HEADER}1,6,h,false,b,f,1,2,3,1,2,3\n{suffix}");
            assert!(
                matches!(read_literal("syntax", &body), Err(ThroughputError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidData),
                "must fail whole file: {suffix:?}"
            );
        }
        for header in ["wrong\n", "schema_version\n", "\n"] {
            assert!(
                matches!(read_literal("header", header), Err(ThroughputError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidData)
            );
        }
    }

    #[test]
    fn csv_00108_append_validates_batch_before_any_write() {
        let path = csv_test_path("invalid-write");
        let good = make_result("good", 1.0, 2.0, 3.0);
        let mut invalid = Vec::new();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            for metric in 0..3 {
                let mut bad = good.clone();
                match metric {
                    0 => bad.stats.p50 = value,
                    1 => bad.stats.p95 = value,
                    _ => bad.stats.p99 = value,
                }
                invalid.push(bad);
            }
        }
        for field in 0..3 {
            let mut bad = good.clone();
            match field {
                0 => bad.timed_frames = 0,
                1 => bad.width = 0,
                _ => bad.height = 0,
            }
            invalid.push(bad);
        }
        for bad in invalid {
            std::fs::write(&path, "unchanged").unwrap();
            let result = append_csv(&path, 7, "h", false, "b", &[good.clone(), bad.clone()]);
            let actual = std::fs::read_to_string(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert!(
                matches!(result, Err(ThroughputError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidInput)
            );
            assert_eq!(
                actual, "unchanged",
                "entire batch must validate before append"
            );
            let absent = path.with_extension("absent-parent");
            assert!(!absent.exists());
            assert!(append_csv(&absent.join("out.csv"), 7, "h", false, "b", &[bad]).is_err());
            assert!(!absent.exists(), "invalid batch must not create parent");
        }
    }

    #[test]
    fn csv_00108_preserves_tiny_positive_metrics() {
        let path = csv_test_path("tiny");
        let _ = std::fs::remove_file(&path);
        append_csv(
            &path,
            7,
            "h",
            false,
            "b",
            &[make_result("tiny", f64::from_bits(1), 0.0001, 0.0004)],
        )
        .unwrap();
        let rows = read_csv(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].p50_us, f64::from_bits(1));
        assert_eq!(rows[0].p95_us, 0.0001);
        assert_eq!(rows[0].p99_us, 0.0004);
    }

    #[test]
    fn csv_00108_ratchet_rejects_invalid_operands_and_product() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            for (base, current, ratio) in [(value, 1.0, 1.1), (1.0, value, 1.1), (1.0, 1.0, value)]
            {
                let msg = check_metric_ratchet("fixture", "p95", base, current, ratio)
                    .expect_err("invalid operands must fail closed");
                assert!(msg.contains("fixture") && msg.contains("p95"));
            }
        }
        assert!(check_metric_ratchet("f", "p50", f64::MAX, 1.0, 2.0).is_err());
        assert!(check_metric_ratchet("f", "p50", f64::from_bits(1), 1.0, 0.1).is_err());
        assert!(check_metric_ratchet("f", "p50", 100.0, 40.0, 0.5).is_ok());
    }

    #[test]
    fn csv_00108_latest_ignores_invalid_public_rows() {
        let valid = CsvRow {
            schema_version: 1,
            timestamp_unix: 1,
            host: "h".into(),
            unpinned: false,
            build_profile_hash: "b".into(),
            fixture: "f".into(),
            p50_us: 1.0,
            p95_us: 2.0,
            p99_us: 3.0,
        };
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            for metric in 0..3 {
                let mut bad = valid.clone();
                bad.timestamp_unix = 2;
                match metric {
                    0 => bad.p50_us = value,
                    1 => bad.p95_us = value,
                    _ => bad.p99_us = value,
                }
                let rows = latest_per_fixture(&[valid.clone(), bad.clone()], &["h"]);
                assert_eq!(
                    rows[0].timestamp_unix, 1,
                    "invalid public row must not replace valid baseline"
                );
                assert!(latest_per_fixture(&[bad], &["h"]).is_empty());
            }
        }
    }

    #[test]
    fn latest_per_fixture_for_build_requires_exact_identity() {
        let make_row = |host: &str,
                        profile: &str,
                        fixture: &str,
                        timestamp_unix: u64,
                        unpinned: bool,
                        p50_us: f64| CsvRow {
            schema_version: CSV_SCHEMA_VERSION,
            timestamp_unix,
            host: host.into(),
            unpinned,
            build_profile_hash: profile.into(),
            fixture: fixture.into(),
            p50_us,
            p95_us: p50_us + 1.0,
            p99_us: p50_us + 2.0,
        };
        let rows = vec![
            make_row("host-a", "profile-a", "z.webm", 10, false, 10.0),
            make_row("host-b", "profile-a", "z.webm", 90, false, 90.0),
            make_row("host-a", "profile-b", "z.webm", 80, false, 80.0),
            make_row("host-a", "profile-a", "z.webm", 70, true, 70.0),
            make_row("host-a", "profile-a", "z.webm", 60, false, f64::NAN),
            make_row("host-a", "profile-a", "z.webm", 20, false, 20.0),
            make_row("host-a", "profile-a", "a.webm", 15, false, 15.0),
            make_row("host-b", "profile-b", "missing.webm", 100, false, 100.0),
        ];

        let latest = latest_per_fixture_for_build(&rows, "host-a", "profile-a");
        assert_eq!(
            latest
                .iter()
                .map(|row| row.fixture.as_str())
                .collect::<Vec<_>>(),
            ["a.webm", "z.webm"],
            "only exact, valid, pinned identities should have baselines"
        );
        let z = latest.iter().find(|row| row.fixture == "z.webm").unwrap();
        assert_eq!(z.timestamp_unix, 20, "newer incompatible rows must not win");
        assert_eq!(z.host, "host-a");
        assert_eq!(z.build_profile_hash, "profile-a");
        assert!(!z.unpinned);

        assert!(
            latest_per_fixture_for_build(&rows, "host-c", "profile-a").is_empty(),
            "no exact identity match must yield no baselines"
        );
    }

    #[test]
    fn malformed_ivf_packet_is_reported_with_fixture_and_index() {
        let path = std::env::temp_dir().join(format!(
            "rvp9-throughput-malformed-packet-{}.ivf",
            std::process::id()
        ));
        let mut ivf = Vec::new();
        write_ivf(&mut ivf, &IvfHeader::vp9(16, 16, 30, 1), &[&[0u8][..]]).unwrap();
        std::fs::write(&path, ivf).unwrap();

        let err = run_one(&path, 0, 1).unwrap_err();
        std::fs::remove_file(&path).unwrap();

        match err {
            ThroughputError::PacketParse {
                fixture,
                packet_index,
                error,
            } => {
                assert_eq!(
                    fixture,
                    path.file_name().unwrap().to_string_lossy(),
                    "fixture basename"
                );
                assert_eq!(packet_index, 1);
                assert!(!error.is_empty());
            }
            other => panic!("expected PacketParse, got {other:?}"),
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
        let msg = check_metric_ratchet("fix.webm", "p99", 100.0, 120.001, 1.20).unwrap_err();
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
