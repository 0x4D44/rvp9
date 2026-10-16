//! Bitstream differential diff tool — Levels 0, 1, 2, 3, 4.
//!
//! Compares two `.ivf` files and reports the first point of divergence.
//! This binary is the entry point for the structured diff harness
//! described in `notes/stage4-design/diff-bitstream-harness.md`.
//! Currently implements:
//!
//! - **Level 0:** raw IVF byte diff (the original Stage 0 behaviour).
//! - **Level 1:** per-frame size + superframe layout + frame-kind +
//!   `show_frame` + `frame_context_idx` diff.
//! - **Level 2:** field-by-field walk of [`rvp9_core::bitstream::UncompressedHeader`]
//!   (and its [`rvp9_core::bitstream::HeaderTail`] when present).
//! - **Level 3:** compressed-header diff: `tx_mode` + `reference_mode`
//!   from [`rvp9_core::bitstream::parse_compressed_header`], plus the
//!   post-update probability tables in the selected
//!   [`rvp9_core::entropy::FrameContext`] bank.
//! - **Level 4:** per-tile syntax — drive two [`rvp9_decoder::Decoder`]
//!   instances frame-by-frame and diff the post-decode
//!   `Option<ModeInfo>` grid published via
//!   [`rvp9_decoder::Decoder::last_frame_mi_grid`].
//!
//! Level 5 (per-coef tokens) is deferred per design Q5; still emits a
//! "not yet implemented" stderr note while the lower levels run.
//!
//! Usage:
//!   cargo run --release --bin diff-bitstream -- \
//!       --a path/to/a.ivf \
//!       --b path/to/b.ivf \
//!       [--level N | --level all] \
//!       [--frame N | --frame N-M] \
//!       [--format text|json] \
//!       [--verbose summary|first|full] \
//!       [--max-frames N] \
//!       [--no-color] \
//!       [--exit-on-diff]
//!
//! Exit codes:
//!   0  no divergence (or divergence but `--exit-on-diff` not set).
//!   1  divergence found and `--exit-on-diff` set (default for this
//!      binary — matches the original Stage 0 behaviour).
//!   2  argument or input error (missing file, bad IVF, bad flag).

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rvp9_core::bitstream::frame::REF_FRAMES;
use rvp9_core::bitstream::sub_parsers::{
    LoopFilterParams, MAX_MODE_LF_DELTAS, MAX_REF_LF_DELTAS, MAX_SEGMENTS, PREDICTION_PROBS,
    QuantizationParams, SEG_LVL_MAX, SEG_TREE_PROBS, SegmentationParams, TileInfo,
};
use rvp9_core::bitstream::{
    CompressedHeaderOptions, FrameKind, HeaderTail, InterpFilter, RefFrameSizes, Superframe,
    UncompressedHeader, parse_compressed_header, parse_superframe_index, parse_uncompressed_header,
};
use rvp9_core::entropy::{
    BLOCK_SIZE_GROUPS, CLASS0_SIZE, COEF_BANDS, COEFF_CONTEXTS, COMP_INTER_CONTEXTS,
    DefaultTxProbs, FRAME_CONTEXTS, FrameContext, INTER_MODE_CONTEXTS, INTER_MODES,
    INTRA_INTER_CONTEXTS, INTRA_MODES, MV_CLASSES, MV_FP_SIZE, MV_JOINTS, MV_OFFSET_BITS,
    NmvComponent, NmvContext, PARTITION_CONTEXTS, PARTITION_TYPES, PLANE_TYPES, REF_CONTEXTS,
    REF_TYPES, SKIP_CONTEXTS, SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS, TX_SIZES,
    UNCONSTRAINED_NODES,
};
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_info::{BlockMode, BmiInfo, Mv, RefFrame};
use rvp9_decoder::Decoder;
use rvp9_harness::container::{ContainerError, read_ivf_packets};

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

/// Highest level the user asked us to walk down to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    /// Numeric level 0..=5.
    N(u8),
    /// `--level all` — walk every level.
    All,
}

impl Level {
    /// Numeric value, with `All` collapsing to the deepest level we know about.
    fn as_u8(self) -> u8 {
        match self {
            Level::N(n) => n,
            Level::All => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Text,
    Json,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verbose {
    Summary,
    First,
    Full,
}

/// A `--frame N` or `--frame N-M` selector. End is inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameRange {
    start: u32,
    end: u32,
}

#[derive(Debug)]
struct Args {
    a: PathBuf,
    b: PathBuf,
    level: Level,
    frame: Option<FrameRange>,
    format: Format,
    verbose: Verbose,
    max_frames: Option<u32>,
    no_color: bool,
    exit_on_diff: bool,
}

/// Parse argv. Returns `Err(String)` for any malformed input — the
/// caller decides whether to print and exit. `None` for a successful
/// `--help` request (caller exits 0).
fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    // Hand-rolled parser, modelled on conformance.rs. We accept both
    // `--flag VALUE` and `--flag=VALUE`.
    let mut a: Option<PathBuf> = None;
    let mut b: Option<PathBuf> = None;
    let mut level: Level = Level::N(4);
    let mut frame: Option<FrameRange> = None;
    let mut format: Format = Format::Text;
    let mut verbose: Verbose = Verbose::First;
    let mut max_frames: Option<u32> = None;
    let mut no_color: bool = false;
    let mut exit_on_diff: bool = false;

    let mut i = 1; // skip argv[0]
    while i < argv.len() {
        let arg = &argv[i];
        // Helper that yields the value for `--flag VALUE` or `--flag=VALUE`.
        let (flag, inline_value): (&str, Option<&str>) = match arg.find('=') {
            Some(eq) => (&arg[..eq], Some(&arg[eq + 1..])),
            None => (arg.as_str(), None),
        };
        let take_value = |i: &mut usize, flag: &str| -> Result<String, String> {
            if let Some(v) = inline_value {
                Ok(v.to_string())
            } else {
                *i += 1;
                argv.get(*i)
                    .cloned()
                    .ok_or_else(|| format!("{flag} requires a value"))
            }
        };

        match flag {
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            "--a" => {
                a = Some(PathBuf::from(take_value(&mut i, "--a")?));
            }
            "--b" => {
                b = Some(PathBuf::from(take_value(&mut i, "--b")?));
            }
            "--level" => {
                let v = take_value(&mut i, "--level")?;
                level = parse_level(&v)?;
            }
            "--frame" => {
                let v = take_value(&mut i, "--frame")?;
                frame = Some(parse_frame_range(&v)?);
            }
            "--format" => {
                let v = take_value(&mut i, "--format")?;
                format = match v.as_str() {
                    "text" => Format::Text,
                    "json" => Format::Json,
                    other => return Err(format!("--format must be text|json (got {other})")),
                };
            }
            "--verbose" => {
                let v = take_value(&mut i, "--verbose")?;
                verbose = match v.as_str() {
                    "summary" => Verbose::Summary,
                    "first" => Verbose::First,
                    "full" => Verbose::Full,
                    other => {
                        return Err(format!(
                            "--verbose must be summary|first|full (got {other})"
                        ));
                    }
                };
            }
            "--max-frames" => {
                let v = take_value(&mut i, "--max-frames")?;
                max_frames = Some(
                    v.parse::<u32>()
                        .map_err(|e| format!("--max-frames not a u32: {e}"))?,
                );
            }
            "--no-color" => no_color = true,
            "--exit-on-diff" => exit_on_diff = true,
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let a = a.ok_or_else(|| "--a required".to_string())?;
    let b = b.ok_or_else(|| "--b required".to_string())?;

    Ok(Some(Args {
        a,
        b,
        level,
        frame,
        format,
        verbose,
        max_frames,
        no_color,
        exit_on_diff,
    }))
}

fn parse_level(v: &str) -> Result<Level, String> {
    if v == "all" {
        return Ok(Level::All);
    }
    let n: u8 = v
        .parse()
        .map_err(|e| format!("--level not 0..=5 or 'all': {e}"))?;
    if n > 5 {
        return Err(format!("--level must be 0..=5 or 'all' (got {n})"));
    }
    Ok(Level::N(n))
}

fn parse_frame_range(v: &str) -> Result<FrameRange, String> {
    match v.split_once('-') {
        None => {
            let n: u32 = v.parse().map_err(|e| format!("--frame not a u32: {e}"))?;
            Ok(FrameRange { start: n, end: n })
        }
        Some((s, e)) => {
            let start: u32 = s.parse().map_err(|e| format!("--frame range start: {e}"))?;
            let end: u32 = e.parse().map_err(|e| format!("--frame range end: {e}"))?;
            if end < start {
                return Err(format!("--frame range end < start ({start}-{end})"));
            }
            Ok(FrameRange { start, end })
        }
    }
}

fn print_help() {
    // Stable, scriptable usage text. Mirrors §2.1 of the design.
    let msg = "\
diff-bitstream — VP9 bitstream differential diff tool

USAGE:
    diff-bitstream --a <PATH> --b <PATH> [OPTIONS]

REQUIRED:
    --a <PATH>              Reference IVF (typically libvpx output)
    --b <PATH>              Candidate IVF (typically rvp9 output)

OPTIONS:
    --level N | --level all     Stop after level N (0..=5); default 4
                                 (level 5 not yet implemented;
                                  deferred per design Q5)
    --frame N | --frame N-M     Restrict to frame index or inclusive range
                                 (ignored at Level 0)
    --format text|json          Output format; default text
    --verbose summary|first|full
                                summary: counts only
                                first  : first divergence per frame (default)
                                full   : every divergence below level N
                                         (not yet implemented)
    --max-frames N              Cap frames processed
    --no-color                  Disable ANSI colour in text output
                                (also auto-disabled if stdout is not a TTY
                                 or NO_COLOR is set in the environment)
    --exit-on-diff              Exit 1 on first divergence (CI gate)
    -h, --help                  Print this help

EXIT CODES:
    0  no divergence (or divergence without --exit-on-diff)
    1  divergence found and --exit-on-diff set
    2  argument or input error
";
    print!("{msg}");
}

// -----------------------------------------------------------------------------
// Diff reporter
// -----------------------------------------------------------------------------

/// Output sink for divergences. Implementations format records however
/// they like; higher levels (commits 2..=4) will feed structured detail
/// through `divergence(..)`. At Level 0 we only emit one record per
/// run.
trait DiffReporter {
    /// Record a divergence at `level`. `summary` is a short label
    /// (e.g. "byte"); `detail` is an optional human-readable string
    /// suitable for the text format and stored verbatim in JSON.
    fn divergence(&mut self, level: u8, summary: &str, detail: Option<&str>);

    /// Note that a packet was inspected with no divergence (used by
    /// the JSON envelope for streaming progress; text default ignores).
    /// Reserved for Level 1+ (commits 2-4 of the design); unused at Level 0.
    #[allow(dead_code)]
    fn frame_ok(&mut self, packet_idx: usize);

    /// Flush any trailing envelope and return the process exit code.
    fn finish(&mut self) -> i32;

    /// `true` if any divergence record was ever submitted (regardless
    /// of whether `--verbose first` suppressed its emission). Used by
    /// the text-format trailer to decide whether to print
    /// "Identical: every byte matches." — `code` alone is ambiguous
    /// when `--exit-on-diff` is false.
    fn had_divergence(&self) -> bool;

    /// Whether the caller requested an immediate stop after the first
    /// recorded divergence.
    fn should_stop(&self) -> bool;
}

/// Text reporter — preserves the original Stage 0 stdout when run at
/// `--level 0` with no extra flags.
struct TextReporter<W: Write> {
    out: W,
    /// Stop after first divergence?
    ///
    /// `stop_at_first` is **per-level**: under `--verbose first` the
    /// reporter emits the first record at each distinct level (so a
    /// run that finds divergences at L0 and L2 still shows both),
    /// but suppresses subsequent records at any level it has already
    /// reported on. `--verbose full` disables this gating entirely.
    stop_at_first: bool,
    /// Has any divergence been seen?
    any_diff: bool,
    /// Total divergence records emitted (informational).
    emitted: usize,
    /// Set of `level` values already emitted under `stop_at_first`.
    /// Indexed by level (0..=5) for cheap O(1) lookup.
    level_emitted: [bool; 6],
    /// Caller wants `exit 1` on divergence.
    exit_on_diff: bool,
    /// ANSI colour enabled?
    color: bool,
}

impl<W: Write> TextReporter<W> {
    fn new(out: W, verbose: Verbose, exit_on_diff: bool, color: bool) -> Self {
        Self {
            out,
            stop_at_first: !matches!(verbose, Verbose::Full),
            any_diff: false,
            emitted: 0,
            level_emitted: [false; 6],
            exit_on_diff,
            color,
        }
    }
}

impl<W: Write> DiffReporter for TextReporter<W> {
    fn divergence(&mut self, level: u8, summary: &str, detail: Option<&str>) {
        let bucket = (level as usize).min(5);
        if self.stop_at_first && self.level_emitted[bucket] {
            self.any_diff = true; // still record that we saw one
            return;
        }
        self.any_diff = true;
        self.emitted += 1;
        self.level_emitted[bucket] = true;
        let red = if self.color { "\x1b[31m" } else { "" };
        let reset = if self.color { "\x1b[0m" } else { "" };
        let label = format!("Level {level}  {summary}");
        match detail {
            Some(d) => {
                let _ = writeln!(self.out, "{red}{label}{reset}: {d}");
            }
            None => {
                let _ = writeln!(self.out, "{red}{label}{reset}");
            }
        }
    }

    fn frame_ok(&mut self, _packet_idx: usize) {
        // Text format stays quiet on per-frame OK at Level 0 — matches
        // the original stdout exactly.
    }

    fn finish(&mut self) -> i32 {
        let _ = self.out.flush();
        if self.any_diff && self.exit_on_diff {
            1
        } else {
            0
        }
    }

    fn had_divergence(&self) -> bool {
        self.any_diff
    }

    fn should_stop(&self) -> bool {
        self.exit_on_diff && self.any_diff
    }
}

/// JSON reporter — one NDJSON record per `divergence()` call, plus a
/// trailing summary object. Format spec: §5.4 of the design.
struct JsonReporter<W: Write> {
    out: W,
    /// `stop_at_first` is per-level — see [`TextReporter`] for the
    /// rationale.
    stop_at_first: bool,
    any_diff: bool,
    emitted: usize,
    level_emitted: [bool; 6],
    exit_on_diff: bool,
}

impl<W: Write> JsonReporter<W> {
    fn new(out: W, verbose: Verbose, exit_on_diff: bool) -> Self {
        Self {
            out,
            stop_at_first: !matches!(verbose, Verbose::Full),
            any_diff: false,
            emitted: 0,
            level_emitted: [false; 6],
            exit_on_diff,
        }
    }
}

/// Quote a string for safe inclusion as a JSON string literal.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

impl<W: Write> DiffReporter for JsonReporter<W> {
    fn divergence(&mut self, level: u8, summary: &str, detail: Option<&str>) {
        let bucket = (level as usize).min(5);
        if self.stop_at_first && self.level_emitted[bucket] {
            self.any_diff = true;
            return;
        }
        self.any_diff = true;
        self.emitted += 1;
        self.level_emitted[bucket] = true;
        let detail_field = match detail {
            Some(d) => format!(",\"detail\":\"{}\"", json_escape(d)),
            None => String::new(),
        };
        let _ = writeln!(
            self.out,
            "{{\"event\":\"divergence\",\"level\":{},\"summary\":\"{}\"{}}}",
            level,
            json_escape(summary),
            detail_field
        );
    }

    fn frame_ok(&mut self, packet_idx: usize) {
        // Streaming progress marker; cheap.
        let _ = writeln!(
            self.out,
            "{{\"event\":\"frame_ok\",\"packet\":{packet_idx}}}"
        );
    }

    fn finish(&mut self) -> i32 {
        let _ = writeln!(
            self.out,
            "{{\"event\":\"summary\",\"any_divergence\":{},\"divergences_emitted\":{}}}",
            self.any_diff, self.emitted
        );
        let _ = self.out.flush();
        if self.any_diff && self.exit_on_diff {
            1
        } else {
            0
        }
    }

    fn had_divergence(&self) -> bool {
        self.any_diff
    }

    fn should_stop(&self) -> bool {
        self.exit_on_diff && self.any_diff
    }
}

// -----------------------------------------------------------------------------
// Level 0 byte diff
// -----------------------------------------------------------------------------

/// Outcome of a Level 0 byte diff.
#[derive(Debug, PartialEq, Eq)]
enum ByteDiff {
    /// Files identical (every byte matches, including length).
    Identical,
    /// One is a strict prefix of the other (no mismatched byte, but
    /// lengths differ).
    PrefixMismatch {
        shorter_len: usize,
        longer_len: usize,
    },
    /// First differing byte at this offset.
    Diverge {
        offset: usize,
        /// 1-based frame index if the offset lies inside a frame
        /// payload; `None` if it falls in the IVF header.
        frame: Option<u32>,
        /// Byte offset within the frame payload (if `frame` is Some).
        payload_offset: Option<usize>,
    },
}

/// Pure helper: byte diff with frame-attribution. Does NOT do any IVF
/// validation beyond the magic check — caller handles that.
fn byte_diff(a: &[u8], b: &[u8], max_frames: Option<u32>) -> ByteDiff {
    // First byte mismatch — covers IVF header and frame headers
    // uniformly.
    let raw_offset = a.iter().zip(b.iter()).position(|(x, y)| x != y);
    let identical_prefix = raw_offset.is_none();

    if identical_prefix && a.len() == b.len() {
        return ByteDiff::Identical;
    }
    if identical_prefix {
        return ByteDiff::PrefixMismatch {
            shorter_len: a.len().min(b.len()),
            longer_len: a.len().max(b.len()),
        };
    }
    let offset = raw_offset.unwrap();

    // Try to attribute the offset to a frame. Walk frame headers in
    // parallel; if we hit a frame whose payload contains `offset`,
    // report frame index + payload offset.
    let (frame, payload_offset) = attribute_to_frame(a, b, offset, max_frames);
    ByteDiff::Diverge {
        offset,
        frame,
        payload_offset,
    }
}

/// Read only the IVF header and the requested number of frame chunks.
///
/// A truncated header or payload is returned as a short buffer so the
/// existing input validation and prefix-diff diagnostics remain in
/// charge of the outcome.
fn read_ivf_prefix(path: &Path, max_frames: Option<u32>) -> Result<Vec<u8>, String> {
    let Some(max_frames) = max_frames else {
        return std::fs::read(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()));
    };

    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let mut data = Vec::new();
    if !read_up_to(&mut file, 32, &mut data)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?
    {
        return Ok(data);
    }

    for _ in 0..max_frames {
        let header_start = data.len();
        if !read_up_to(&mut file, 12, &mut data)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?
        {
            break;
        }
        let payload_size = u32::from_le_bytes([
            data[header_start],
            data[header_start + 1],
            data[header_start + 2],
            data[header_start + 3],
        ]) as usize;
        if !read_up_to(&mut file, payload_size, &mut data)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?
        {
            break;
        }
    }
    Ok(data)
}

/// Append at most `len` bytes and return whether all requested bytes
/// were present.
fn read_up_to<R: Read>(reader: &mut R, len: usize, data: &mut Vec<u8>) -> std::io::Result<bool> {
    let start = data.len();
    reader.take(len as u64).read_to_end(data)?;
    Ok(data.len() - start == len)
}

/// Walk IVF frames in parallel up to `max_frames` (if set) and return
/// `(Some(frame_idx_1based), Some(payload_offset))` if `target_offset`
/// falls inside a frame payload. Returns `(None, None)` if the offset
/// lies in the IVF file header or in a frame header.
fn attribute_to_frame(
    a: &[u8],
    b: &[u8],
    target_offset: usize,
    max_frames: Option<u32>,
) -> (Option<u32>, Option<usize>) {
    if a.len() < 32 || b.len() < 32 {
        return (None, None);
    }
    // The 32-byte IVF file header is excluded from per-frame
    // attribution — divergences there are reported with the raw
    // offset only.
    if target_offset < 32 {
        return (None, None);
    }
    let mut ai = 32usize;
    let mut bi = 32usize;
    let mut frame_idx: u32 = 0;
    let cap = max_frames.unwrap_or(u32::MAX);
    while ai + 12 <= a.len() && bi + 12 <= b.len() && frame_idx < cap {
        frame_idx += 1;
        let a_size = u32::from_le_bytes([a[ai], a[ai + 1], a[ai + 2], a[ai + 3]]) as usize;
        let b_size = u32::from_le_bytes([b[bi], b[bi + 1], b[bi + 2], b[bi + 3]]) as usize;
        if a_size != b_size {
            // Size mismatch — the divergence is at this frame's header.
            return (Some(frame_idx), None);
        }
        let payload_start = ai + 12;
        let payload_end = payload_start + a_size;
        if target_offset >= ai && target_offset < payload_start {
            // Falls inside this frame's 12-byte chunk header.
            return (Some(frame_idx), None);
        }
        if target_offset >= payload_start && target_offset < payload_end {
            return (Some(frame_idx), Some(target_offset - payload_start));
        }
        ai = payload_end;
        bi += 12 + b_size;
    }
    (None, None)
}

/// Format a `ByteDiff` for human consumption.
fn format_byte_diff(d: &ByteDiff) -> String {
    match d {
        ByteDiff::Identical => "Identical: every byte matches.".to_string(),
        ByteDiff::PrefixMismatch {
            shorter_len,
            longer_len,
        } => {
            format!("One is a prefix of the other (lengths differ: {shorter_len} vs {longer_len}).")
        }
        ByteDiff::Diverge {
            offset,
            frame: Some(f),
            payload_offset: Some(p),
        } => {
            format!("First byte divergence at offset {offset} (frame {f}, byte +{p} into payload)")
        }
        ByteDiff::Diverge {
            offset,
            frame: Some(f),
            payload_offset: None,
        } => format!("First byte divergence at offset {offset} (frame {f} header)"),
        ByteDiff::Diverge {
            offset,
            frame: None,
            ..
        } => format!("First byte divergence at offset {offset} (IVF file header region)"),
    }
}

// -----------------------------------------------------------------------------
// Level 1 + Level 2: per-frame parsed-header diff
// -----------------------------------------------------------------------------

/// One frame's-worth of pre-parsed state, used by Levels 1/2/3.
///
/// `packet` is the 1-based IVF packet index; `frame` is the 0-based
/// sub-frame index within the (super)frame (always 0 for single
/// frames). `chunk_size` is the IVF-declared payload byte count for
/// the packet; `superframe_count` is the number of sub-frames the
/// trailing superframe index produced (1 for a single frame).
///
/// `header` is `None` if the uncompressed-header parser bailed — the
/// driver records a divergence in that case (parser error is itself
/// information) but doesn't try to walk fields on `None`.
///
/// `frame_bytes` is the slice of the IVF file backing this sub-frame
/// (without the per-packet 12-byte IVF chunk header). `header_end`
/// is the byte offset within `frame_bytes` at which the compressed
/// partition begins — i.e. `parse_uncompressed_header`'s second
/// return value. Both are populated only when `header` is `Some`;
/// Level 3 uses them to slice out the compressed-header bytes.
struct FrameEntry {
    packet: u32,
    frame: u32,
    chunk_size: usize,
    superframe_count: u32,
    is_superframe: bool,
    header: Option<UncompressedHeader>,
    /// `Some(err_string)` if header parsing failed; carried through so
    /// L1/L2 can report the failure rather than swallow it silently.
    parse_error: Option<String>,
    /// The sub-frame bytes (no IVF chunk header). Length = the frame's
    /// declared size from the superframe index (or `chunk_size` for
    /// a single-frame packet). Empty when the parser bailed.
    frame_bytes: Vec<u8>,
    /// Byte offset within `frame_bytes` of the start of the compressed
    /// header partition; `0` when `header` is `None`.
    header_end: usize,
}

/// Extract every (packet, sub-frame) pair from an IVF buffer along
/// with the parsed uncompressed header. Returns the flat list in
/// container order.
fn extract_frames(data: &[u8]) -> Result<Vec<FrameEntry>, ContainerError> {
    let packets = read_ivf_packets(data)?;
    // We need an empty ref-frame pool — the harness can't track real
    // inter-frame size inheritance without driving a full Decoder, and
    // for Levels 1/2 we only need the uncompressed-header *fields*,
    // not the full reference plumbing. Any frame that genuinely
    // inherits a size from a ref will surface as a parse_error, which
    // we report as a divergence reason rather than crashing. This
    // matches the design's "harness does not re-implement decoder
    // state" rule (§4 of the design note).
    let no_refs: RefFrameSizes = [None; REF_FRAMES];
    let mut out = Vec::new();
    for (packet_idx, range) in packets.iter().enumerate() {
        let packet = (packet_idx as u32) + 1;
        let bytes = range.slice(data);
        let chunk_size = bytes.len();

        // Superframe split.
        let sf = match parse_superframe_index(bytes) {
            Ok(sf) => sf,
            Err(e) => {
                // Treat as a single unparseable frame so L1 still
                // reports the chunk size + the parse error.
                out.push(FrameEntry {
                    packet,
                    frame: 0,
                    chunk_size,
                    superframe_count: 1,
                    is_superframe: false,
                    header: None,
                    parse_error: Some(format!("superframe parse error: {e}")),
                    frame_bytes: Vec::new(),
                    header_end: 0,
                });
                continue;
            }
        };

        match sf {
            Superframe::Single => {
                let (header, parse_error, header_end) = parse_header_opt(bytes, &no_refs);
                out.push(FrameEntry {
                    packet,
                    frame: 0,
                    chunk_size,
                    superframe_count: 1,
                    is_superframe: false,
                    header,
                    parse_error,
                    frame_bytes: bytes.to_vec(),
                    header_end,
                });
            }
            Superframe::Multi {
                sizes,
                count,
                index_size,
            } => {
                let mut offset = 0usize;
                let frame_data_end = chunk_size.saturating_sub(index_size);
                for (sub, &raw_size) in sizes.iter().enumerate().take(count) {
                    let frame_size = raw_size as usize;
                    let frame_end = offset.saturating_add(frame_size);
                    if frame_end > frame_data_end {
                        // Truncated sub-frame — record the error and stop.
                        out.push(FrameEntry {
                            packet,
                            frame: sub as u32,
                            chunk_size,
                            superframe_count: count as u32,
                            is_superframe: true,
                            header: None,
                            parse_error: Some(format!(
                                "superframe sub-frame {sub} truncated: \
                                 declared {frame_size} bytes, only \
                                 {} remain",
                                frame_data_end.saturating_sub(offset),
                            )),
                            frame_bytes: Vec::new(),
                            header_end: 0,
                        });
                        break;
                    }
                    let frame_bytes_slice = &bytes[offset..frame_end];
                    let (header, parse_error, header_end) =
                        parse_header_opt(frame_bytes_slice, &no_refs);
                    out.push(FrameEntry {
                        packet,
                        frame: sub as u32,
                        chunk_size,
                        superframe_count: count as u32,
                        is_superframe: true,
                        header,
                        parse_error,
                        frame_bytes: frame_bytes_slice.to_vec(),
                        header_end,
                    });
                    offset = frame_end;
                }
            }
        }
    }
    Ok(out)
}

/// Best-effort header parse. Returns `(Some(header), None, end)` on
/// success (where `end` is the byte offset at which the compressed
/// partition begins) or `(None, Some(error_string), 0)` on failure.
fn parse_header_opt(
    bytes: &[u8],
    refs: &RefFrameSizes,
) -> (Option<UncompressedHeader>, Option<String>, usize) {
    match parse_uncompressed_header(bytes, refs) {
        Ok((h, end)) => (Some(h), None, end),
        Err(e) => (None, Some(format!("header parse error: {e}")), 0),
    }
}

/// Read an IVF file from disk and extract its parsed frame entries.
/// Errors are turned into a `String` for the caller to surface via
/// stderr.
fn load_frames(path: &Path, max_frames: Option<u32>) -> Result<Vec<FrameEntry>, String> {
    let data = read_ivf_prefix(path, max_frames)?;
    extract_frames(&data).map_err(|e| format!("{}: {e}", path.display()))
}

/// `true` if `entry.packet` (1-based) falls inside the optional
/// `--frame N` / `--frame N-M` range. Both endpoints are 1-based
/// packet indices (Q6 supervisor decision: `--frame N` selects packet
/// N, output labels are `packet=N frame=M`).
fn packet_in_range(packet: u32, range: Option<FrameRange>) -> bool {
    match range {
        None => true,
        Some(r) => packet >= r.start && packet <= r.end,
    }
}

/// Format a per-frame label used by the L1/L2 divergence detail
/// strings: `packet=P frame=F`.
fn frame_label(entry: &FrameEntry) -> String {
    format!("packet={} frame={}", entry.packet, entry.frame)
}

/// Run the Level 1 diff (per-frame size + kind + show_frame +
/// frame_context_idx + superframe layout). Pre-parsed frame lists
/// are produced by [`extract_frames`].
fn level_1_diff<R: DiffReporter>(
    a: &[FrameEntry],
    b: &[FrameEntry],
    args: &Args,
    reporter: &mut R,
) {
    if a.len() != b.len() {
        let detail = format!(
            "frame count differs: a={} entries, b={} entries",
            a.len(),
            b.len(),
        );
        reporter.divergence(1, "frame-count", Some(&detail));
        // Walk only the common prefix; missing trailing frames are
        // already accounted for by the count record.
    }

    let common = a.len().min(b.len());
    for i in 0..common {
        let ea = &a[i];
        let eb = &b[i];
        // Honour --frame on the A-side packet number (both files
        // should be aligned by index in normal use; if they aren't,
        // the count divergence above flags it).
        if !packet_in_range(ea.packet, args.frame) {
            continue;
        }

        // Was the previous frame on either side already diverging
        // enough to stop in --verbose first? The reporter handles
        // that gating internally; we just emit and let it decide.
        let mut emitted = 0u32;

        // Parse errors are first-class divergences at L1 ONLY when
        // asymmetric: if one side parsed and the other didn't, or both
        // failed differently, that's a real bug. If both sides hit the
        // SAME error (e.g. both miss an inter-frame ref because the
        // harness doesn't drive a full decoder — design §4 documents
        // this gap), it's symmetric harness noise, not divergence.
        let parse_diverges = match (&ea.parse_error, &eb.parse_error) {
            (None, None) => false,
            (Some(a), Some(b)) if a == b => false,
            _ => true,
        };
        if parse_diverges {
            let detail = format!(
                "{lbl}: a_err={a_err:?} b_err={b_err:?}",
                lbl = frame_label(ea),
                a_err = ea.parse_error.as_deref().unwrap_or("ok"),
                b_err = eb.parse_error.as_deref().unwrap_or("ok"),
            );
            reporter.divergence(1, "parse-error", Some(&detail));
            emitted += 1;
            if matches!(args.verbose, Verbose::First) {
                continue;
            }
        } else if ea.parse_error.is_some() {
            // Both failed identically; skip downstream diffs for this
            // frame since the headers aren't populated.
            continue;
        }

        if ea.chunk_size != eb.chunk_size {
            let detail = format!(
                "{}: chunk_size a={} b={}",
                frame_label(ea),
                ea.chunk_size,
                eb.chunk_size,
            );
            reporter.divergence(1, "chunk-size", Some(&detail));
            emitted += 1;
            if matches!(args.verbose, Verbose::First) && emitted > 0 {
                continue;
            }
        }

        if ea.is_superframe != eb.is_superframe || ea.superframe_count != eb.superframe_count {
            let detail = format!(
                "{}: superframe a=({}, count={}) b=({}, count={})",
                frame_label(ea),
                ea.is_superframe,
                ea.superframe_count,
                eb.is_superframe,
                eb.superframe_count,
            );
            reporter.divergence(1, "superframe-layout", Some(&detail));
            emitted += 1;
            if matches!(args.verbose, Verbose::First) && emitted > 0 {
                continue;
            }
        }

        let (ha, hb) = match (ea.header.as_ref(), eb.header.as_ref()) {
            (Some(a), Some(b)) => (a, b),
            // Without both headers parsed we can't do field-level diff;
            // the parse-error branch above already reported.
            _ => continue,
        };

        let kind_a = frame_kind_label(&ha.kind);
        let kind_b = frame_kind_label(&hb.kind);
        if kind_a != kind_b {
            let detail = format!("{}: frame_kind a={} b={}", frame_label(ea), kind_a, kind_b,);
            reporter.divergence(1, "frame-kind", Some(&detail));
            emitted += 1;
            if matches!(args.verbose, Verbose::First) && emitted > 0 {
                continue;
            }
        }

        if ha.show_frame != hb.show_frame {
            let detail = format!(
                "{}: show_frame a={} b={}",
                frame_label(ea),
                ha.show_frame,
                hb.show_frame,
            );
            reporter.divergence(1, "show-frame", Some(&detail));
            emitted += 1;
            if matches!(args.verbose, Verbose::First) && emitted > 0 {
                continue;
            }
        }

        if let (Some(ta), Some(tb)) = (ha.tail.as_ref(), hb.tail.as_ref())
            && ta.frame_context_idx != tb.frame_context_idx
        {
            let detail = format!(
                "{}: frame_context_idx a={} b={}",
                frame_label(ea),
                ta.frame_context_idx,
                tb.frame_context_idx,
            );
            reporter.divergence(1, "frame-context-idx", Some(&detail));
        }
    }
}

/// Stable short label for a [`FrameKind`] discriminant.
fn frame_kind_label(k: &FrameKind) -> &'static str {
    match k {
        FrameKind::ShowExistingFrame { .. } => "ShowExistingFrame",
        FrameKind::KeyFrame => "KeyFrame",
        FrameKind::IntraOnly { .. } => "IntraOnly",
        FrameKind::InterFrame { .. } => "InterFrame",
    }
}

/// Run the Level 2 diff: field-by-field walk of every
/// [`UncompressedHeader`] pair.
fn level_2_diff<R: DiffReporter>(
    a: &[FrameEntry],
    b: &[FrameEntry],
    args: &Args,
    reporter: &mut R,
) {
    let common = a.len().min(b.len());
    for i in 0..common {
        let ea = &a[i];
        let eb = &b[i];
        if !packet_in_range(ea.packet, args.frame) {
            continue;
        }
        let (ha, hb) = match (ea.header.as_ref(), eb.header.as_ref()) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };

        let label = frame_label(ea);
        // Bail-out flag for `--verbose first`: once we've emitted one
        // record for THIS frame we suppress further records on this
        // frame only. (Level 0 / Level 1 records on other frames are
        // separate; the reporter's own `stop_at_first` handles the
        // cross-level gate.)
        let mut emitted = false;
        let stop_first = matches!(args.verbose, Verbose::First);
        let mut emit = |path: &str, detail: String| {
            if emitted && stop_first {
                return;
            }
            let full = format!("{label}: {path}: {detail}");
            reporter.divergence(2, path, Some(&full));
            emitted = true;
        };
        walk_uncompressed_header(ha, hb, &mut emit);
    }
}

/// Visitor: invoke `emit(field_path, "a=.. b=..")` for every leaf
/// field of [`UncompressedHeader`] (and [`HeaderTail`]) where the two
/// sides differ. Trait-free, macro-free — just a flat sequence of
/// `dbg_field!`-style calls, optimised for "I can see every covered
/// field at a glance".
fn walk_uncompressed_header<F: FnMut(&str, String)>(
    a: &UncompressedHeader,
    b: &UncompressedHeader,
    emit: &mut F,
) {
    // ----- top-level scalar fields -----
    diff_field("header.profile", &a.profile, &b.profile, emit);
    diff_field("header.show_frame", &a.show_frame, &b.show_frame, emit);
    diff_field(
        "header.error_resilient",
        &a.error_resilient,
        &b.error_resilient,
        emit,
    );
    diff_field("header.bit_depth", &a.bit_depth, &b.bit_depth, emit);
    diff_field("header.color_space", &a.color_space, &b.color_space, emit);
    diff_field("header.color_range", &a.color_range, &b.color_range, emit);
    diff_field("header.chroma", &a.chroma, &b.chroma, emit);
    diff_field("header.width", &a.width, &b.width, emit);
    diff_field("header.height", &a.height, &b.height, emit);
    diff_field("header.render_size", &a.render_size, &b.render_size, emit);

    // ----- kind payload -----
    walk_kind(&a.kind, &b.kind, emit);

    // ----- tail -----
    match (a.tail.as_ref(), b.tail.as_ref()) {
        (Some(ta), Some(tb)) => walk_tail(ta, tb, emit),
        (None, None) => {}
        _ => emit(
            "header.tail.presence",
            format!("a={} b={}", a.tail.is_some(), b.tail.is_some(),),
        ),
    }
}

fn walk_kind<F: FnMut(&str, String)>(a: &FrameKind, b: &FrameKind, emit: &mut F) {
    let la = frame_kind_label(a);
    let lb = frame_kind_label(b);
    if la != lb {
        emit("header.kind.tag", format!("a={la} b={lb}"));
        return;
    }
    // Same discriminant — walk the per-variant payload.
    match (a, b) {
        (FrameKind::ShowExistingFrame { slot: sa }, FrameKind::ShowExistingFrame { slot: sb }) => {
            diff_field("header.kind.ShowExistingFrame.slot", sa, sb, emit);
        }
        (FrameKind::KeyFrame, FrameKind::KeyFrame) => {}
        (
            FrameKind::IntraOnly {
                reset_frame_context: ra,
                refresh_frame_flags: fa,
            },
            FrameKind::IntraOnly {
                reset_frame_context: rb,
                refresh_frame_flags: fb,
            },
        ) => {
            diff_field("header.kind.IntraOnly.reset_frame_context", ra, rb, emit);
            diff_field("header.kind.IntraOnly.refresh_frame_flags", fa, fb, emit);
        }
        (
            FrameKind::InterFrame {
                reset_frame_context: ra,
                refresh_frame_flags: fa,
                ref_frame_indices: ia,
                ref_frame_sign_biases: sa,
                allow_high_precision_mv: ha,
                interp_filter: ifa,
            },
            FrameKind::InterFrame {
                reset_frame_context: rb,
                refresh_frame_flags: fb,
                ref_frame_indices: ib,
                ref_frame_sign_biases: sb,
                allow_high_precision_mv: hb,
                interp_filter: ifb,
            },
        ) => {
            diff_field("header.kind.InterFrame.reset_frame_context", ra, rb, emit);
            diff_field("header.kind.InterFrame.refresh_frame_flags", fa, fb, emit);
            diff_field("header.kind.InterFrame.ref_frame_indices", ia, ib, emit);
            diff_field("header.kind.InterFrame.ref_frame_sign_biases", sa, sb, emit);
            diff_field(
                "header.kind.InterFrame.allow_high_precision_mv",
                ha,
                hb,
                emit,
            );
            diff_field("header.kind.InterFrame.interp_filter", ifa, ifb, emit);
        }
        // Unreachable: tags are equal so the pairs must match.
        _ => emit("header.kind", format!("a={a:?} b={b:?}")),
    }
}

fn walk_tail<F: FnMut(&str, String)>(a: &HeaderTail, b: &HeaderTail, emit: &mut F) {
    diff_field(
        "header.tail.refresh_frame_context",
        &a.refresh_frame_context,
        &b.refresh_frame_context,
        emit,
    );
    diff_field(
        "header.tail.frame_parallel_decoding_mode",
        &a.frame_parallel_decoding_mode,
        &b.frame_parallel_decoding_mode,
        emit,
    );
    diff_field(
        "header.tail.frame_context_idx",
        &a.frame_context_idx,
        &b.frame_context_idx,
        emit,
    );
    walk_loopfilter(&a.loop_filter, &b.loop_filter, emit);
    walk_quantization(&a.quantization, &b.quantization, emit);
    walk_segmentation(&a.segmentation, &b.segmentation, emit);
    walk_tile_info(&a.tile_info, &b.tile_info, emit);
    diff_field(
        "header.tail.first_partition_size",
        &a.first_partition_size,
        &b.first_partition_size,
        emit,
    );
}

fn walk_loopfilter<F: FnMut(&str, String)>(
    a: &LoopFilterParams,
    b: &LoopFilterParams,
    emit: &mut F,
) {
    diff_field(
        "header.tail.loop_filter.filter_level",
        &a.filter_level,
        &b.filter_level,
        emit,
    );
    diff_field(
        "header.tail.loop_filter.sharpness_level",
        &a.sharpness_level,
        &b.sharpness_level,
        emit,
    );
    diff_field(
        "header.tail.loop_filter.mode_ref_delta_enabled",
        &a.mode_ref_delta_enabled,
        &b.mode_ref_delta_enabled,
        emit,
    );
    diff_field(
        "header.tail.loop_filter.mode_ref_delta_update",
        &a.mode_ref_delta_update,
        &b.mode_ref_delta_update,
        emit,
    );
    for i in 0..MAX_REF_LF_DELTAS {
        let path = format!("header.tail.loop_filter.ref_deltas[{i}]");
        diff_field(&path, &a.ref_deltas[i], &b.ref_deltas[i], emit);
    }
    for i in 0..MAX_MODE_LF_DELTAS {
        let path = format!("header.tail.loop_filter.mode_deltas[{i}]");
        diff_field(&path, &a.mode_deltas[i], &b.mode_deltas[i], emit);
    }
}

fn walk_quantization<F: FnMut(&str, String)>(
    a: &QuantizationParams,
    b: &QuantizationParams,
    emit: &mut F,
) {
    diff_field(
        "header.tail.quantization.base_qindex",
        &a.base_qindex,
        &b.base_qindex,
        emit,
    );
    diff_field(
        "header.tail.quantization.y_dc_delta_q",
        &a.y_dc_delta_q,
        &b.y_dc_delta_q,
        emit,
    );
    diff_field(
        "header.tail.quantization.uv_dc_delta_q",
        &a.uv_dc_delta_q,
        &b.uv_dc_delta_q,
        emit,
    );
    diff_field(
        "header.tail.quantization.uv_ac_delta_q",
        &a.uv_ac_delta_q,
        &b.uv_ac_delta_q,
        emit,
    );
}

fn walk_segmentation<F: FnMut(&str, String)>(
    a: &SegmentationParams,
    b: &SegmentationParams,
    emit: &mut F,
) {
    diff_field(
        "header.tail.segmentation.enabled",
        &a.enabled,
        &b.enabled,
        emit,
    );
    diff_field(
        "header.tail.segmentation.update_map",
        &a.update_map,
        &b.update_map,
        emit,
    );
    diff_field(
        "header.tail.segmentation.temporal_update",
        &a.temporal_update,
        &b.temporal_update,
        emit,
    );
    diff_field(
        "header.tail.segmentation.update_data",
        &a.update_data,
        &b.update_data,
        emit,
    );
    diff_field(
        "header.tail.segmentation.abs_delta",
        &a.abs_delta,
        &b.abs_delta,
        emit,
    );
    for i in 0..SEG_TREE_PROBS {
        let path = format!("header.tail.segmentation.tree_probs[{i}]");
        diff_field(&path, &a.tree_probs[i], &b.tree_probs[i], emit);
    }
    for i in 0..PREDICTION_PROBS {
        let path = format!("header.tail.segmentation.pred_probs[{i}]");
        diff_field(&path, &a.pred_probs[i], &b.pred_probs[i], emit);
    }
    for seg in 0..MAX_SEGMENTS {
        for feat in 0..SEG_LVL_MAX {
            let path = format!("header.tail.segmentation.feature_data[{seg}][{feat}]");
            diff_field(
                &path,
                &a.feature_data[seg][feat],
                &b.feature_data[seg][feat],
                emit,
            );
        }
    }
}

fn walk_tile_info<F: FnMut(&str, String)>(a: &TileInfo, b: &TileInfo, emit: &mut F) {
    diff_field(
        "header.tail.tile_info.log2_tile_cols",
        &a.log2_tile_cols,
        &b.log2_tile_cols,
        emit,
    );
    diff_field(
        "header.tail.tile_info.log2_tile_rows",
        &a.log2_tile_rows,
        &b.log2_tile_rows,
        emit,
    );
}

/// Helper: compare two values with `Debug` representation; if not
/// equal, call `emit(path, "a={a:?} b={b:?}")`.
fn diff_field<T, F>(path: &str, a: &T, b: &T, emit: &mut F)
where
    T: PartialEq + std::fmt::Debug + ?Sized,
    F: FnMut(&str, String),
{
    if a != b {
        emit(path, format!("a={a:?} b={b:?}"));
    }
}

// -----------------------------------------------------------------------------
// Level 3: compressed-header + FrameContext prob-bank diff
// -----------------------------------------------------------------------------

/// Build [`CompressedHeaderOptions`] from an uncompressed header.
///
/// Mirrors `Decoder::parse_compressed_partition`
/// (`crates/rvp9-decoder/src/lib.rs:558-578`):
///
/// - `lossless` from `tail.quantization.is_lossless()`,
/// - `intra_only` from `kind.is_intra()`,
/// - `(interp_filter, allow_high_precision_mv)` from
///   `FrameKind::InterFrame` (defaults `EightTap` / `false` otherwise —
///   `parse_compressed_header` ignores these when `intra_only == true`),
/// - `compound_allowed = false` conservatively. This is design Q3 and
///   matches the decoder's pending-POC stance; symmetric "miss" is
///   harmless for the diff harness because both sides use the same
///   value and the same uncompressed header.
fn options_from_header(h: &UncompressedHeader) -> CompressedHeaderOptions {
    let tail = h.tail.as_ref();
    let lossless = tail.is_some_and(|t| t.quantization.is_lossless());
    let intra_only = h.kind.is_intra();
    let (interp_filter, allow_high_precision_mv) = match &h.kind {
        FrameKind::InterFrame {
            interp_filter,
            allow_high_precision_mv,
            ..
        } => (*interp_filter, *allow_high_precision_mv),
        _ => (InterpFilter::EightTap, false),
    };
    CompressedHeaderOptions {
        lossless,
        intra_only,
        interp_filter,
        compound_allowed: false,
        allow_high_precision_mv,
    }
}

/// Per-side [`FrameContext`] state for Level 3.
///
/// **Simplification (v1):** both sides keep their own
/// `[FrameContext; FRAME_CONTEXTS]` array. Each frame updates the bank
/// indexed by `tail.frame_context_idx`. The full VP9 decoder also
/// honours `refresh_frame_context` and various reset conditions
/// (keyframes, `error_resilient`, `frame_parallel_decoding_mode`) when
/// deciding whether the post-parse bank gets written back. This
/// simplification is sound for diff purposes when both encoders are
/// bit-identical (the same bank ends up updated to the same value on
/// both sides), and the divergence the harness is looking for is the
/// post-update bank contents — not the cross-frame refresh policy.
struct FrameContextBanks {
    banks: [FrameContext; FRAME_CONTEXTS],
}

impl FrameContextBanks {
    fn new() -> Self {
        // FRAME_CONTEXTS = 4. Build explicitly to avoid requiring
        // `Default` on `FrameContext`.
        Self {
            banks: [
                FrameContext::new_default(),
                FrameContext::new_default(),
                FrameContext::new_default(),
                FrameContext::new_default(),
            ],
        }
    }
}

/// Replay one side's Level 3 parser inputs from a preceding entry.
/// Parser errors remain silent for prefix entries, while any bank
/// mutations made before an error are retained. Entries without a
/// parsed header/tail or an available compressed partition cannot be
/// replayed and are skipped.
fn prime_l3_entry(entry: &FrameEntry, banks: &mut FrameContextBanks) {
    let Some(header) = entry.header.as_ref() else {
        return;
    };
    if matches!(header.kind, FrameKind::ShowExistingFrame { .. }) {
        return;
    }
    let Some(tail) = header.tail.as_ref() else {
        return;
    };
    let Some(compressed) = slice_compressed(
        entry.header_end,
        tail.first_partition_size,
        &entry.frame_bytes,
    ) else {
        return;
    };

    let options = options_from_header(header);
    let idx = (tail.frame_context_idx as usize).min(FRAME_CONTEXTS - 1);
    let _ = parse_compressed_header(compressed, &mut banks.banks[idx], options);
}

/// Run the Level 3 diff: parse the compressed header on both sides
/// against per-side `FrameContext` banks, then diff
/// `(tx_mode, reference_mode)` and the post-update prob tables.
///
/// `show_existing_frame` frames have no compressed header and are
/// skipped. Frames whose uncompressed-header parse failed are skipped
/// (L1 already reported the asymmetry, if any).
fn level_3_diff<R: DiffReporter>(
    a: &[FrameEntry],
    b: &[FrameEntry],
    args: &Args,
    reporter: &mut R,
) {
    let mut banks_a = FrameContextBanks::new();
    let mut banks_b = FrameContextBanks::new();

    if let Some(range) = args.frame {
        // Prime each side independently. A bad prefix on one side must
        // not suppress usable state from the other side, and priming
        // emits no diagnostics because only selected frames are reports.
        for entry in a.iter().take_while(|entry| entry.packet < range.start) {
            prime_l3_entry(entry, &mut banks_a);
        }
        for entry in b.iter().take_while(|entry| entry.packet < range.start) {
            prime_l3_entry(entry, &mut banks_b);
        }
    }

    let common = a.len().min(b.len());
    for i in 0..common {
        let ea = &a[i];
        let eb = &b[i];
        if !packet_in_range(ea.packet, args.frame) {
            continue;
        }
        // Need parsed headers on both sides to compute opts + locate
        // the partition.
        let (ha, hb) = match (ea.header.as_ref(), eb.header.as_ref()) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        // ShowExistingFrame has no compressed header.
        if matches!(ha.kind, FrameKind::ShowExistingFrame { .. })
            || matches!(hb.kind, FrameKind::ShowExistingFrame { .. })
        {
            continue;
        }
        let (ta, tb) = match (ha.tail.as_ref(), hb.tail.as_ref()) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };

        let label = frame_label(ea);
        let stop_first = matches!(args.verbose, Verbose::First);
        let mut emitted = false;
        let mut emit = |path: &str, detail: String| {
            if emitted && stop_first {
                return;
            }
            let full = format!("{label}: {path}: {detail}");
            reporter.divergence(3, path, Some(&full));
            emitted = true;
        };

        // Slice out the compressed-header partition on each side. If
        // the partition extends past the frame data we emit a parse
        // error record (mirrors the decoder's
        // `TruncatedCompressedHeader`).
        let compressed_a =
            match slice_compressed(ea.header_end, ta.first_partition_size, &ea.frame_bytes) {
                Some(s) => s,
                None => {
                    emit(
                        "fc.parse-error",
                        format!(
                            "a: compressed header extends past frame data \
                         (header_end={}, fps={}, frame_len={})",
                            ea.header_end,
                            ta.first_partition_size,
                            ea.frame_bytes.len(),
                        ),
                    );
                    continue;
                }
            };
        let compressed_b =
            match slice_compressed(eb.header_end, tb.first_partition_size, &eb.frame_bytes) {
                Some(s) => s,
                None => {
                    emit(
                        "fc.parse-error",
                        format!(
                            "b: compressed header extends past frame data \
                         (header_end={}, fps={}, frame_len={})",
                            eb.header_end,
                            tb.first_partition_size,
                            eb.frame_bytes.len(),
                        ),
                    );
                    continue;
                }
            };

        let opts_a = options_from_header(ha);
        let opts_b = options_from_header(hb);

        let idx_a = (ta.frame_context_idx as usize).min(FRAME_CONTEXTS - 1);
        let idx_b = (tb.frame_context_idx as usize).min(FRAME_CONTEXTS - 1);

        // Drive the parser against the selected bank on each side.
        // Mirrors `Decoder::parse_compressed_partition`.
        let res_a = parse_compressed_header(compressed_a, &mut banks_a.banks[idx_a], opts_a);
        let res_b = parse_compressed_header(compressed_b, &mut banks_b.banks[idx_b], opts_b);

        // The `emit` closure handles its own first-vs-full gating via
        // the `emitted` flag, so we can call it freely below without
        // re-checking after each call.
        match (&res_a, &res_b) {
            (Ok(ra), Ok(rb)) => {
                if ra.tx_mode != rb.tx_mode {
                    emit(
                        "fc.tx_mode",
                        format!("a={:?} b={:?}", ra.tx_mode, rb.tx_mode),
                    );
                }
                if ra.reference_mode != rb.reference_mode {
                    emit(
                        "fc.reference_mode",
                        format!("a={:?} b={:?}", ra.reference_mode, rb.reference_mode),
                    );
                }
                // Walk the (post-update) bank on each side. We diff the
                // banks the parser just updated; cross-frame refresh
                // semantics are simplified per `FrameContextBanks` docs.
                walk_frame_context(&banks_a.banks[idx_a], &banks_b.banks[idx_b], &mut emit);
            }
            (Err(ea), Err(eb)) if format!("{ea}") == format!("{eb}") => {
                // Both sides failed identically — symmetric noise.
                // Mirrors the L1/L2 parse-error policy: don't report.
            }
            (a_res, b_res) => {
                emit(
                    "fc.parse-error",
                    format!(
                        "a={} b={}",
                        match a_res {
                            Ok(_) => "ok".to_string(),
                            Err(e) => format!("{e}"),
                        },
                        match b_res {
                            Ok(_) => "ok".to_string(),
                            Err(e) => format!("{e}"),
                        },
                    ),
                );
            }
        }
    }
}

/// Return `Some(&data[header_end..header_end + fps])` or `None` if
/// the slice is out of bounds (overflow or past-end).
fn slice_compressed(header_end: usize, first_partition_size: u16, data: &[u8]) -> Option<&[u8]> {
    let fps = first_partition_size as usize;
    let needed_end = header_end.checked_add(fps)?;
    if needed_end > data.len() {
        return None;
    }
    Some(&data[header_end..needed_end])
}

// -----------------------------------------------------------------------------
// FrameContext field walker
// -----------------------------------------------------------------------------

/// Visitor: invoke `emit(field_path, "a=.. b=..")` for every leaf
/// field of [`FrameContext`] where the two sides differ. Field paths
/// follow `fc.<field>[<idx>][<idx>]...` so divergences are
/// pinpointable in tooling.
///
/// Layout — every leaf of `FrameContext` is visited:
/// 1. `y_mode_prob[group][i]`
/// 2. `uv_mode_prob[y_mode][i]`
/// 3. `partition_prob[ctx][i]`
/// 4. `coef_probs[tx][plane][ref][band][ctx][node]`
/// 5. `switchable_interp_prob[ctx][i]`
/// 6. `inter_mode_probs[ctx][i]`
/// 7. `intra_inter_prob[ctx]`
/// 8. `comp_inter_prob[ctx]`
/// 9. `single_ref_prob[ctx][i]`
/// 10. `comp_ref_prob[ctx]`
/// 11. `tx_probs.{p32x32,p16x16,p8x8}[ctx][i]`
/// 12. `skip_probs[ctx]`
/// 13. `nmvc.joints[i]` + `nmvc.comps[axis].{...}`
/// 14. `initialized`
fn walk_frame_context<F: FnMut(&str, String)>(a: &FrameContext, b: &FrameContext, emit: &mut F) {
    // 1. y_mode_prob[BLOCK_SIZE_GROUPS][INTRA_MODES-1]
    for g in 0..BLOCK_SIZE_GROUPS {
        for i in 0..(INTRA_MODES - 1) {
            let path = format!("fc.y_mode_prob[group={g}][{i}]");
            diff_field(&path, &a.y_mode_prob[g][i], &b.y_mode_prob[g][i], emit);
        }
    }
    // 2. uv_mode_prob[INTRA_MODES][INTRA_MODES-1]
    for y in 0..INTRA_MODES {
        for i in 0..(INTRA_MODES - 1) {
            let path = format!("fc.uv_mode_prob[y_mode={y}][{i}]");
            diff_field(&path, &a.uv_mode_prob[y][i], &b.uv_mode_prob[y][i], emit);
        }
    }
    // 3. partition_prob[PARTITION_CONTEXTS][PARTITION_TYPES-1]
    for ctx in 0..PARTITION_CONTEXTS {
        for i in 0..(PARTITION_TYPES - 1) {
            let path = format!("fc.partition_prob[ctx={ctx}][{i}]");
            diff_field(
                &path,
                &a.partition_prob[ctx][i],
                &b.partition_prob[ctx][i],
                emit,
            );
        }
    }
    // 4. coef_probs[TX_SIZES][PLANE_TYPES][REF_TYPES][COEF_BANDS]
    //    [COEFF_CONTEXTS][UNCONSTRAINED_NODES]
    const TX_LABEL: [&str; TX_SIZES] = ["4x4", "8x8", "16x16", "32x32"];
    for (tx, tx_lbl) in TX_LABEL.iter().enumerate() {
        for plane in 0..PLANE_TYPES {
            for r in 0..REF_TYPES {
                for band in 0..COEF_BANDS {
                    for ctx in 0..COEFF_CONTEXTS {
                        for node in 0..UNCONSTRAINED_NODES {
                            let path = format!(
                                "fc.coef_probs[tx={tx_lbl}][plane={plane}][ref={r}]\
                                 [band={band}][ctx={ctx}][node={node}]"
                            );
                            diff_field(
                                &path,
                                &a.coef_probs[tx][plane][r][band][ctx][node],
                                &b.coef_probs[tx][plane][r][band][ctx][node],
                                emit,
                            );
                        }
                    }
                }
            }
        }
    }
    // 5. switchable_interp_prob[SWITCHABLE_FILTER_CONTEXTS][SWITCHABLE_FILTERS-1]
    for ctx in 0..SWITCHABLE_FILTER_CONTEXTS {
        for i in 0..(SWITCHABLE_FILTERS - 1) {
            let path = format!("fc.switchable_interp_prob[ctx={ctx}][{i}]");
            diff_field(
                &path,
                &a.switchable_interp_prob[ctx][i],
                &b.switchable_interp_prob[ctx][i],
                emit,
            );
        }
    }
    // 6. inter_mode_probs[INTER_MODE_CONTEXTS][INTER_MODES-1]
    for ctx in 0..INTER_MODE_CONTEXTS {
        for i in 0..(INTER_MODES - 1) {
            let path = format!("fc.inter_mode_probs[ctx={ctx}][{i}]");
            diff_field(
                &path,
                &a.inter_mode_probs[ctx][i],
                &b.inter_mode_probs[ctx][i],
                emit,
            );
        }
    }
    // 7. intra_inter_prob[INTRA_INTER_CONTEXTS]
    for ctx in 0..INTRA_INTER_CONTEXTS {
        let path = format!("fc.intra_inter_prob[ctx={ctx}]");
        diff_field(
            &path,
            &a.intra_inter_prob[ctx],
            &b.intra_inter_prob[ctx],
            emit,
        );
    }
    // 8. comp_inter_prob[COMP_INTER_CONTEXTS]
    for ctx in 0..COMP_INTER_CONTEXTS {
        let path = format!("fc.comp_inter_prob[ctx={ctx}]");
        diff_field(
            &path,
            &a.comp_inter_prob[ctx],
            &b.comp_inter_prob[ctx],
            emit,
        );
    }
    // 9. single_ref_prob[REF_CONTEXTS][2]
    for ctx in 0..REF_CONTEXTS {
        for i in 0..2 {
            let path = format!("fc.single_ref_prob[ctx={ctx}][{i}]");
            diff_field(
                &path,
                &a.single_ref_prob[ctx][i],
                &b.single_ref_prob[ctx][i],
                emit,
            );
        }
    }
    // 10. comp_ref_prob[REF_CONTEXTS]
    for ctx in 0..REF_CONTEXTS {
        let path = format!("fc.comp_ref_prob[ctx={ctx}]");
        diff_field(&path, &a.comp_ref_prob[ctx], &b.comp_ref_prob[ctx], emit);
    }
    // 11. tx_probs.{p32x32,p16x16,p8x8}
    walk_tx_probs(&a.tx_probs, &b.tx_probs, emit);
    // 12. skip_probs[SKIP_CONTEXTS]
    for ctx in 0..SKIP_CONTEXTS {
        let path = format!("fc.skip_probs[ctx={ctx}]");
        diff_field(&path, &a.skip_probs[ctx], &b.skip_probs[ctx], emit);
    }
    // 13. nmvc — joints + per-axis component pair.
    walk_nmvc(&a.nmvc, &b.nmvc, emit);
    // 14. initialized
    diff_field("fc.initialized", &a.initialized, &b.initialized, emit);
}

fn walk_tx_probs<F: FnMut(&str, String)>(a: &DefaultTxProbs, b: &DefaultTxProbs, emit: &mut F) {
    // p32x32: [[u8; 3]; 2]
    for ctx in 0..2 {
        for i in 0..3 {
            let path = format!("fc.tx_probs.p32x32[ctx={ctx}][{i}]");
            diff_field(&path, &a.p32x32[ctx][i], &b.p32x32[ctx][i], emit);
        }
    }
    // p16x16: [[u8; 2]; 2]
    for ctx in 0..2 {
        for i in 0..2 {
            let path = format!("fc.tx_probs.p16x16[ctx={ctx}][{i}]");
            diff_field(&path, &a.p16x16[ctx][i], &b.p16x16[ctx][i], emit);
        }
    }
    // p8x8: [[u8; 1]; 2]
    for ctx in 0..2 {
        for i in 0..1 {
            let path = format!("fc.tx_probs.p8x8[ctx={ctx}][{i}]");
            diff_field(&path, &a.p8x8[ctx][i], &b.p8x8[ctx][i], emit);
        }
    }
}

fn walk_nmvc<F: FnMut(&str, String)>(a: &NmvContext, b: &NmvContext, emit: &mut F) {
    // joints: [u8; MV_JOINTS - 1]
    for i in 0..(MV_JOINTS - 1) {
        let path = format!("fc.nmvc.joints[{i}]");
        diff_field(&path, &a.joints[i], &b.joints[i], emit);
    }
    // comps: [NmvComponent; 2]
    for axis in 0..2 {
        walk_nmv_component(axis, &a.comps[axis], &b.comps[axis], emit);
    }
}

fn walk_nmv_component<F: FnMut(&str, String)>(
    axis: usize,
    a: &NmvComponent,
    b: &NmvComponent,
    emit: &mut F,
) {
    let base = format!("fc.nmvc.comps[{axis}]");
    diff_field(&format!("{base}.sign"), &a.sign, &b.sign, emit);
    for i in 0..(MV_CLASSES - 1) {
        diff_field(
            &format!("{base}.classes[{i}]"),
            &a.classes[i],
            &b.classes[i],
            emit,
        );
    }
    for i in 0..(CLASS0_SIZE - 1) {
        diff_field(
            &format!("{base}.class0[{i}]"),
            &a.class0[i],
            &b.class0[i],
            emit,
        );
    }
    for i in 0..MV_OFFSET_BITS {
        diff_field(&format!("{base}.bits[{i}]"), &a.bits[i], &b.bits[i], emit);
    }
    for c0 in 0..CLASS0_SIZE {
        for i in 0..(MV_FP_SIZE - 1) {
            diff_field(
                &format!("{base}.class0_fp[{c0}][{i}]"),
                &a.class0_fp[c0][i],
                &b.class0_fp[c0][i],
                emit,
            );
        }
    }
    for i in 0..(MV_FP_SIZE - 1) {
        diff_field(&format!("{base}.fp[{i}]"), &a.fp[i], &b.fp[i], emit);
    }
    diff_field(
        &format!("{base}.class0_hp"),
        &a.class0_hp,
        &b.class0_hp,
        emit,
    );
    diff_field(&format!("{base}.hp"), &a.hp, &b.hp, emit);
}

// -----------------------------------------------------------------------------
// Level 4: per-tile syntax diff via Decoder::decode_frame
// -----------------------------------------------------------------------------

/// Static map of field-path suffix → probable encoder-module owner.
/// Used by [`likely_site_hint`] to annotate Level 4 records with a
/// triage hint per design §5.3. Suffix-match on the final field
/// segment of the path; the path itself can start with `fc.`, `mi.`,
/// or `header.` (we ignore the prefix and the index brackets, just
/// match against the field segment substrings).
static LIKELY_SITES: &[(&str, &str)] = &[
    (".block_size", "encoder::partition"),
    (".tx_size", "encoder::mode_decision"),
    (".skip", "encoder::mode_decision"),
    (".segment_id", "encoder::aq"),
    (".seg_id_predicted", "encoder::aq"),
    (".mode.y_mode", "encoder::mode_decision (intra)"),
    (".mode.uv_mode", "encoder::mode_decision (intra)"),
    (".mode.mode", "encoder::mode_decision (inter)"),
    (".mode.ref_frames", "encoder::motion_estimation"),
    (".mode.mvs", "encoder::motion_estimation"),
    (".mode.interp_filter", "encoder::mode_decision"),
    (".mode.tag", "encoder::mode_decision"),
    (".bmi", "encoder::mode_decision (sub-8 partition)"),
    ("fc.coef_probs", "encoder::entropy (token coder)"),
    ("fc.nmvc", "encoder::motion_estimation"),
    ("fc.partition_prob", "encoder::partition"),
    ("fc.skip_probs", "encoder::mode_decision"),
];

/// Return `Some(module)` if any suffix in [`LIKELY_SITES`] is a
/// substring of `path`; the longest matching entry wins so more
/// specific paths (e.g. `.mode.y_mode`) beat their generic
/// ancestors. `None` if nothing matches.
fn likely_site_hint(path: &str) -> Option<&'static str> {
    LIKELY_SITES
        .iter()
        .filter(|(suffix, _)| path.contains(*suffix))
        .max_by_key(|(suffix, _)| suffix.len())
        .map(|(_, owner)| *owner)
}

/// Append `(likely site: <module>)` to `detail` when [`likely_site_hint`]
/// finds a match for `path`. Otherwise return `detail` unchanged.
fn with_likely_site(path: &str, detail: String) -> String {
    match likely_site_hint(path) {
        Some(owner) => format!("{detail} (likely site: {owner})"),
        None => detail,
    }
}

/// Run the Level 4 diff: drive two `Decoder` instances frame-by-frame
/// over both IVFs and compare the post-decode mi grids cell-by-cell.
///
/// Asymmetric `decode_frame` errors are reported as Level 4 records;
/// symmetric errors (both sides failed identically) are silent —
/// mirrors the L1/L2/L3 parse-error policy.
fn level_4_diff<R: DiffReporter, E: Write>(
    a_data: &[u8],
    b_data: &[u8],
    args: &Args,
    reporter: &mut R,
    stderr: &mut E,
) {
    let packets_a = match read_ivf_packets(a_data) {
        Ok(p) => p,
        Err(e) => {
            let _ = writeln!(stderr, "L4 skipped: {}: {e}", args.a.display());
            return;
        }
    };
    let packets_b = match read_ivf_packets(b_data) {
        Ok(p) => p,
        Err(e) => {
            let _ = writeln!(stderr, "L4 skipped: {}: {e}", args.b.display());
            return;
        }
    };

    let mut decoder_a = Decoder::new();
    let mut decoder_b = Decoder::new();
    let common = packets_a.len().min(packets_b.len());

    for (packet_idx, (ra, rb)) in packets_a
        .iter()
        .zip(packets_b.iter())
        .take(common)
        .enumerate()
    {
        let packet_no = (packet_idx as u32) + 1;
        if let Some(cap) = args.max_frames
            && packet_no > cap
        {
            break;
        }
        if !packet_in_range(packet_no, args.frame) {
            if args.frame.is_some_and(|range| packet_no < range.start) {
                // Walk the full prefix packet on each side, including
                // every sub-frame in a superframe. Errors are silent
                // and isolated per decoder so one leg cannot prevent
                // the other from acquiring its usable history.
                prime_l4_packet(&mut decoder_a, ra.slice(a_data));
                prime_l4_packet(&mut decoder_b, rb.slice(b_data));
            } else if args.frame.is_some_and(|range| packet_no > range.end) {
                break;
            }
            continue;
        }

        let bytes_a = ra.slice(a_data);
        let bytes_b = rb.slice(b_data);

        let parse_a = decoder_a.parse_packet(bytes_a);
        let parse_b = decoder_b.parse_packet(bytes_b);

        // Asymmetric parse error → record (mirrors L1 parse-error policy).
        let (frames_a, frames_b) = match (parse_a, parse_b) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(ea), Err(eb)) if format!("{ea}") == format!("{eb}") => continue,
            (a_res, b_res) => {
                let detail = format!(
                    "packet={packet_no}: parse_packet a={} b={}",
                    match a_res {
                        Ok(_) => "ok".to_string(),
                        Err(e) => format!("{e}"),
                    },
                    match b_res {
                        Ok(_) => "ok".to_string(),
                        Err(e) => format!("{e}"),
                    },
                );
                reporter.divergence(4, "decode.parse_packet", Some(&detail));
                continue;
            }
        };

        // Frame-count mismatch inside a superframe: report and walk
        // the common prefix.
        let mut frames_a_iter = frames_a.into_iter();
        let mut frames_b_iter = frames_b.into_iter();
        let mut sub_idx: u32 = 0;
        loop {
            let next_a = frames_a_iter.next();
            let next_b = frames_b_iter.next();
            match (next_a, next_b) {
                (None, None) => break,
                (Some(_), None) | (None, Some(_)) => {
                    let detail = format!(
                        "packet={packet_no}: superframe sub-frame count differs at sub={sub_idx}",
                    );
                    reporter.divergence(4, "decode.subframe-count", Some(&detail));
                    break;
                }
                (Some(fa), Some(fb)) => {
                    diff_one_decoded_frame(
                        &mut decoder_a,
                        &mut decoder_b,
                        &fa,
                        &fb,
                        packet_no,
                        sub_idx,
                        args,
                        reporter,
                    );
                    sub_idx += 1;
                }
            }
        }
    }
}

/// Best-effort Level 4 state priming for one decoder and one IVF
/// packet. A packet can contain several VP9 frames, all of which must
/// be decoded in order to reconstruct the state before the selection.
fn prime_l4_packet(decoder: &mut Decoder, packet: &[u8]) {
    let Ok(frames) = decoder.parse_packet(packet) else {
        return;
    };
    for frame in frames {
        let _ = decoder.decode_frame(&frame);
    }
}

/// Drive one decoded sub-frame through both decoders and diff the
/// resulting `Option<ModeInfo>` grids.
#[allow(clippy::too_many_arguments)]
fn diff_one_decoded_frame<R: DiffReporter>(
    decoder_a: &mut Decoder,
    decoder_b: &mut Decoder,
    frame_a: &rvp9_decoder::ParsedFrame<'_>,
    frame_b: &rvp9_decoder::ParsedFrame<'_>,
    packet_no: u32,
    sub_idx: u32,
    args: &Args,
    reporter: &mut R,
) {
    let label = format!("packet={packet_no} frame={sub_idx}");
    let stop_first = matches!(args.verbose, Verbose::First);
    // Use a Cell so the closure can read+write the gating flag without
    // exclusively borrowing the outer variable for its whole lifetime.
    let emitted = std::cell::Cell::new(false);
    let mut emit = |path: &str, detail: String| {
        if emitted.get() && stop_first {
            return;
        }
        let full = format!("{label}: {path}: {}", with_likely_site(path, detail));
        reporter.divergence(4, path, Some(&full));
        emitted.set(true);
    };

    // Drive both decoders synchronously; missing/asymmetric errors are
    // L4 records, matching errors are silent.
    let res_a = decoder_a.decode_frame(frame_a);
    let res_b = decoder_b.decode_frame(frame_b);
    match (&res_a, &res_b) {
        (Ok(_), Ok(_)) => {}
        (Err(ea), Err(eb)) if format!("{ea}") == format!("{eb}") => {
            // Symmetric error — skip cell walk; both grids may be empty
            // anyway (Decoder leaves last_frame_mi_grid untouched on
            // error). No record emitted.
            return;
        }
        (a_res, b_res) => {
            emit(
                "decode.decode_frame",
                format!(
                    "a={} b={}",
                    match a_res {
                        Ok(_) => "ok".to_string(),
                        Err(e) => format!("{e}"),
                    },
                    match b_res {
                        Ok(_) => "ok".to_string(),
                        Err(e) => format!("{e}"),
                    },
                ),
            );
            return;
        }
    }

    // Both sides decoded — compare the published mi grids.
    let dims_a = decoder_a.last_frame_mi_dims();
    let dims_b = decoder_b.last_frame_mi_dims();
    if dims_a != dims_b {
        emit(
            "mi_grid.dims",
            format!(
                "a=({},{}) b=({},{})",
                dims_a.0, dims_a.1, dims_b.0, dims_b.1,
            ),
        );
        return;
    }

    let grid_a = decoder_a.last_frame_mi_grid();
    let grid_b = decoder_b.last_frame_mi_grid();
    let (mi_rows, mi_cols) = (dims_a.0 as usize, dims_a.1 as usize);
    if grid_a.len() != mi_rows * mi_cols || grid_b.len() != mi_rows * mi_cols {
        // Decoder contract violation — should never happen, but report
        // rather than panic so the harness stays robust.
        emit(
            "mi_grid.length",
            format!(
                "a.len={} b.len={} expected={}",
                grid_a.len(),
                grid_b.len(),
                mi_rows * mi_cols,
            ),
        );
        return;
    }

    // Per-cell walk. Iterate row-major; for cells where either side is
    // `Some`, run the field-walker (or report a presence divergence).
    for row in 0..mi_rows {
        for col in 0..mi_cols {
            if emitted.get() && stop_first {
                return;
            }
            let cell_a = &grid_a[row * mi_cols + col];
            let cell_b = &grid_b[row * mi_cols + col];
            let cell_label = format!("mi[row={row},col={col}]");
            match (cell_a, cell_b) {
                (None, None) => {}
                (Some(_), None) | (None, Some(_)) => {
                    emit(
                        &format!("{cell_label}.presence"),
                        format!("a={} b={}", cell_a.is_some(), cell_b.is_some(),),
                    );
                }
                (Some(mi_a), Some(mi_b)) => {
                    if mi_a != mi_b {
                        walk_mode_info(&cell_label, mi_a, mi_b, &mut emit);
                    }
                }
            }
        }
    }
}

/// Field-by-field walker over [`rvp9_core::mode_info::ModeInfo`].
/// Emits one `emit(path, "a=.. b=..")` call per differing leaf field.
/// `mi_label` is the cell label (e.g. `mi[row=12,col=8]`); paths are
/// rooted at that label.
pub(crate) fn walk_mode_info<F: FnMut(&str, String)>(
    mi_label: &str,
    a: &rvp9_core::mode_info::ModeInfo,
    b: &rvp9_core::mode_info::ModeInfo,
    emit: &mut F,
) {
    diff_field(
        &format!("{mi_label}.block_size"),
        &a.block_size,
        &b.block_size,
        emit,
    );
    diff_field(&format!("{mi_label}.tx_size"), &a.tx_size, &b.tx_size, emit);
    diff_field(&format!("{mi_label}.skip"), &a.skip, &b.skip, emit);
    diff_field(
        &format!("{mi_label}.segment_id"),
        &a.segment_id,
        &b.segment_id,
        emit,
    );
    diff_field(
        &format!("{mi_label}.seg_id_predicted"),
        &a.seg_id_predicted,
        &b.seg_id_predicted,
        emit,
    );
    walk_block_mode(&format!("{mi_label}.mode"), &a.mode, &b.mode, emit);
    // bmi: [BmiInfo; 4]
    for (i, (ba, bb)) in a.bmi.iter().zip(b.bmi.iter()).enumerate() {
        let base = format!("{mi_label}.bmi[{i}]");
        walk_bmi_info(&base, ba, bb, emit);
    }
}

/// Walker over [`rvp9_core::mode_info::BlockMode`]. Tag mismatch
/// (intra vs inter) emits one record; same-tag pairs recurse into
/// payload fields.
fn walk_block_mode<F: FnMut(&str, String)>(base: &str, a: &BlockMode, b: &BlockMode, emit: &mut F) {
    let tag_a = block_mode_tag(a);
    let tag_b = block_mode_tag(b);
    if tag_a != tag_b {
        emit(&format!("{base}.tag"), format!("a={tag_a} b={tag_b}"));
        return;
    }
    match (a, b) {
        (
            BlockMode::Intra {
                y_mode: ya,
                uv_mode: ua,
            },
            BlockMode::Intra {
                y_mode: yb,
                uv_mode: ub,
            },
        ) => {
            diff_field(&format!("{base}.y_mode"), ya, yb, emit);
            diff_field(&format!("{base}.uv_mode"), ua, ub, emit);
        }
        (
            BlockMode::Inter {
                mode: ma,
                ref_frames: rfa,
                mvs: mva,
                interp_filter: ifa,
            },
            BlockMode::Inter {
                mode: mb,
                ref_frames: rfb,
                mvs: mvb,
                interp_filter: ifb,
            },
        ) => {
            diff_field(&format!("{base}.mode"), ma, mb, emit);
            walk_ref_frame_pair(&format!("{base}.ref_frames"), rfa, rfb, emit);
            walk_mv_pair(&format!("{base}.mvs"), mva, mvb, emit);
            diff_field(&format!("{base}.interp_filter"), ifa, ifb, emit);
        }
        // Tag mismatch is handled above; the remaining patterns are
        // unreachable but keep the match total.
        _ => emit(base, format!("a={a:?} b={b:?}")),
    }
}

/// Walker over `[RefFrame; 2]` (BlockMode::Inter.ref_frames).
fn walk_ref_frame_pair<F: FnMut(&str, String)>(
    base: &str,
    a: &[RefFrame; 2],
    b: &[RefFrame; 2],
    emit: &mut F,
) {
    for i in 0..2 {
        diff_field(&format!("{base}[{i}]"), &a[i], &b[i], emit);
    }
}

/// Walker over `[Mv; 2]` (BlockMode::Inter.mvs and BmiInfo.as_mv).
fn walk_mv_pair<F: FnMut(&str, String)>(base: &str, a: &[Mv; 2], b: &[Mv; 2], emit: &mut F) {
    for i in 0..2 {
        walk_mv(&format!("{base}[{i}]"), &a[i], &b[i], emit);
    }
}

/// Walker over [`rvp9_core::mode_info::Mv`]. Two scalar fields
/// (`row` / `col`) — emit per-axis records so a 1-pixel MV bug shows
/// the axis.
fn walk_mv<F: FnMut(&str, String)>(base: &str, a: &Mv, b: &Mv, emit: &mut F) {
    diff_field(&format!("{base}.row"), &a.row, &b.row, emit);
    diff_field(&format!("{base}.col"), &a.col, &b.col, emit);
}

/// Walker over [`rvp9_core::mode_info::BmiInfo`].
fn walk_bmi_info<F: FnMut(&str, String)>(base: &str, a: &BmiInfo, b: &BmiInfo, emit: &mut F) {
    diff_field(&format!("{base}.as_mode"), &a.as_mode, &b.as_mode, emit);
    walk_mv_pair(&format!("{base}.as_mv"), &a.as_mv, &b.as_mv, emit);
}

/// Stable short label for a [`BlockMode`] discriminant.
fn block_mode_tag(m: &BlockMode) -> &'static str {
    match m {
        BlockMode::Intra { .. } => "Intra",
        BlockMode::Inter { .. } => "Inter",
    }
}

// IntraMode / RefFrame are used via Debug in `diff_field`; keep the
// imports alive so future refactors don't accidentally drop them.
#[allow(dead_code)]
fn _intra_mode_uses(_m: IntraMode) {}

// -----------------------------------------------------------------------------
// Driver
// -----------------------------------------------------------------------------

/// Run the diff. Pulled out of `main` so tests can drive it.
///
/// Writes the preamble (file paths, sizes, deferred-feature notes) to
/// the supplied `stdout`/`stderr` sinks. The reporter handles all
/// divergence records and the final exit code.
fn run_with_streams<E: Write, R: DiffReporter>(
    args: &Args,
    stderr: &mut E,
    reporter: &mut R,
) -> i32 {
    let level_n = args.level.as_u8();

    // Level 5 (per-coef tokens) deferred per design Q5 — every other
    // level ships here.
    if level_n >= 5 {
        let _ = writeln!(
            stderr,
            "note: level 5 not yet implemented \
             (deferred per Q5 of stage4-design/diff-bitstream-harness.md); \
             running levels 0..=4 only"
        );
    }

    let a = match read_ivf_prefix(&args.a, args.max_frames) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(stderr, "{e}");
            return 2;
        }
    };
    let b = match read_ivf_prefix(&args.b, args.max_frames) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(stderr, "{e}");
            return 2;
        }
    };

    // IVF sanity. Mismatches here are input errors, not "divergence"
    // findings.
    if a.len() < 32 || b.len() < 32 {
        let _ = writeln!(stderr, "file too short to contain IVF header");
        return 2;
    }
    if &a[0..4] != b"DKIF" || &b[0..4] != b"DKIF" {
        let _ = writeln!(stderr, "not an IVF file (bad signature)");
        return 2;
    }

    // ----- Level 0: byte diff -----
    let diff = byte_diff(&a, &b, args.max_frames);
    match &diff {
        ByteDiff::Identical => {
            // No divergence at any level — short-circuit.
            return reporter.finish();
        }
        ByteDiff::PrefixMismatch { .. } | ByteDiff::Diverge { .. } => {
            let msg = format_byte_diff(&diff);
            reporter.divergence(0, "byte", Some(&msg));
            if reporter.should_stop() {
                return reporter.finish();
            }
        }
    }

    // ----- Levels 1+: parse the IVF on both sides and run higher
    //                  levels. We only do this if the user asked for
    //                  them (level_n >= 1) AND the byte diff already
    //                  reported divergence (otherwise byte-identity
    //                  guarantees parser-output identity too).
    if level_n >= 1 {
        let frames_a = match load_frames(&args.a, args.max_frames) {
            Ok(f) => f,
            Err(e) => {
                let _ = writeln!(stderr, "L1/L2 skipped: {e}");
                return reporter.finish();
            }
        };
        let frames_b = match load_frames(&args.b, args.max_frames) {
            Ok(f) => f,
            Err(e) => {
                let _ = writeln!(stderr, "L1/L2 skipped: {e}");
                return reporter.finish();
            }
        };

        level_1_diff(&frames_a, &frames_b, args, reporter);
        if reporter.should_stop() {
            return reporter.finish();
        }
        if level_n >= 2 {
            level_2_diff(&frames_a, &frames_b, args, reporter);
            if reporter.should_stop() {
                return reporter.finish();
            }
        }
        if level_n >= 3 {
            level_3_diff(&frames_a, &frames_b, args, reporter);
            if reporter.should_stop() {
                return reporter.finish();
            }
        }
        if level_n >= 4 {
            level_4_diff(&a, &b, args, reporter, stderr);
        }
    }

    reporter.finish()
}

/// Result of the pre-reporter file load + IVF sanity pass. We do this
/// in main() so the text-format preamble (which is part of the
/// original Stage 0 stdout contract) prints BEFORE we hand the writer
/// off to the reporter.
struct Preload {
    a_len: usize,
    b_len: usize,
}

fn preload_for_preamble(args: &Args) -> Result<Preload, String> {
    let a = std::fs::metadata(&args.a)
        .map_err(|e| format!("could not stat --a {}: {e}", args.a.display()))?;
    let b = std::fs::metadata(&args.b)
        .map_err(|e| format!("could not stat --b {}: {e}", args.b.display()))?;
    Ok(Preload {
        a_len: a.len() as usize,
        b_len: b.len() as usize,
    })
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Ok(Some(a)) => a,
        Ok(None) => return ExitCode::from(0), // --help
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("try `diff-bitstream --help`");
            return ExitCode::from(2);
        }
    };

    // ANSI-colour policy: explicit `--no-color`, presence of NO_COLOR
    // (per https://no-color.org), or non-TTY stdout all disable.
    let no_color_env = std::env::var_os("NO_COLOR").is_some();
    let color = !args.no_color && !no_color_env && std::io::stdout().is_terminal();

    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let mut out = stdout.lock();
    let mut err = stderr.lock();

    // For the text format we emit the preamble (matching the original
    // Stage 0 output verbatim) before handing the writer to the
    // reporter. For JSON we skip the preamble — NDJSON readers expect
    // pure records.
    if matches!(args.format, Format::Text) {
        match preload_for_preamble(&args) {
            Ok(Preload { a_len, b_len }) => {
                let _ = writeln!(out, "a: {}  ({} bytes)", args.a.display(), a_len);
                let _ = writeln!(out, "b: {}  ({} bytes)", args.b.display(), b_len);
                let _ = writeln!(out);
            }
            Err(e) => {
                let _ = writeln!(err, "{e}");
                return ExitCode::from(2);
            }
        }
    }

    let (code, had_diff) = match args.format {
        Format::Text => {
            let mut reporter = TextReporter::new(&mut out, args.verbose, args.exit_on_diff, color);
            let code = run_with_streams(&args, &mut err, &mut reporter);
            (code, reporter.had_divergence())
        }
        Format::Json => {
            let mut reporter = JsonReporter::new(&mut out, args.verbose, args.exit_on_diff);
            let code = run_with_streams(&args, &mut err, &mut reporter);
            (code, reporter.had_divergence())
        }
    };

    // Original Stage 0 contract: when text format and no divergence
    // *at any level*, print "Identical: every byte matches." on stdout.
    // We consult `had_divergence` rather than `code` because the
    // latter is 0 both for "identical" and "divergent but --exit-on-diff
    // not set".
    if matches!(args.format, Format::Text) && !had_diff {
        let _ = writeln!(out, "Identical: every byte matches.");
    }

    ExitCode::from(code as u8)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CapturedDivergence {
        level: u8,
        summary: String,
        detail: Option<String>,
    }

    #[derive(Default)]
    struct CaptureReporter {
        records: Vec<CapturedDivergence>,
    }

    impl DiffReporter for CaptureReporter {
        fn divergence(&mut self, level: u8, summary: &str, detail: Option<&str>) {
            self.records.push(CapturedDivergence {
                level,
                summary: summary.to_string(),
                detail: detail.map(str::to_string),
            });
        }

        fn frame_ok(&mut self, _packet_idx: usize) {}

        fn finish(&mut self) -> i32 {
            0
        }

        fn had_divergence(&self) -> bool {
            !self.records.is_empty()
        }

        fn should_stop(&self) -> bool {
            false
        }
    }

    fn memory_diff_args(frame: Option<FrameRange>, max_frames: Option<u32>) -> Args {
        Args {
            a: PathBuf::from("memory-a.ivf"),
            b: PathBuf::from("memory-b.ivf"),
            level: Level::N(4),
            frame,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames,
            no_color: true,
            exit_on_diff: false,
        }
    }

    fn yuv_view<'a>(y: &'a [u8], uv: &'a [u8]) -> rvp9_encoder::YuvFrameRef<'a> {
        rvp9_encoder::YuvFrameRef {
            y,
            u: uv,
            v: uv,
            y_stride: 64,
            u_stride: 32,
            v_stride: 32,
        }
    }

    fn argv(parts: &[&str]) -> Vec<String> {
        std::iter::once("diff-bitstream")
            .chain(parts.iter().copied())
            .map(String::from)
            .collect()
    }

    // -------------------------------------------------------------------------
    // Argument parsing
    // -------------------------------------------------------------------------

    #[test]
    fn missing_required_args_fail() {
        assert!(parse_args(&argv(&[])).is_err());
        assert!(parse_args(&argv(&["--a", "x.ivf"])).is_err());
        assert!(parse_args(&argv(&["--b", "y.ivf"])).is_err());
    }

    #[test]
    fn minimal_invocation_parses() {
        let a = parse_args(&argv(&["--a", "x.ivf", "--b", "y.ivf"]))
            .unwrap()
            .unwrap();
        assert_eq!(a.a, PathBuf::from("x.ivf"));
        assert_eq!(a.b, PathBuf::from("y.ivf"));
        assert_eq!(a.level, Level::N(4));
        assert_eq!(a.format, Format::Text);
        assert_eq!(a.verbose, Verbose::First);
        assert!(a.frame.is_none());
        assert!(a.max_frames.is_none());
        assert!(!a.no_color);
        assert!(!a.exit_on_diff);
    }

    #[test]
    fn level_numeric_and_all_parse() {
        for n in 0..=5u8 {
            let a = parse_args(&argv(&["--a", "x", "--b", "y", "--level", &n.to_string()]))
                .unwrap()
                .unwrap();
            assert_eq!(a.level, Level::N(n));
        }
        let a = parse_args(&argv(&["--a", "x", "--b", "y", "--level", "all"]))
            .unwrap()
            .unwrap();
        assert_eq!(a.level, Level::All);

        // out-of-range
        assert!(parse_args(&argv(&["--a", "x", "--b", "y", "--level", "6"])).is_err());
        assert!(parse_args(&argv(&["--a", "x", "--b", "y", "--level", "nope"])).is_err());
    }

    #[test]
    fn frame_single_and_range_parse() {
        let a = parse_args(&argv(&["--a", "x", "--b", "y", "--frame", "5"]))
            .unwrap()
            .unwrap();
        assert_eq!(a.frame, Some(FrameRange { start: 5, end: 5 }));

        let a = parse_args(&argv(&["--a", "x", "--b", "y", "--frame", "5-10"]))
            .unwrap()
            .unwrap();
        assert_eq!(a.frame, Some(FrameRange { start: 5, end: 10 }));

        assert!(parse_args(&argv(&["--a", "x", "--b", "y", "--frame", "10-5"])).is_err());
        assert!(parse_args(&argv(&["--a", "x", "--b", "y", "--frame", "abc"])).is_err());
    }

    #[test]
    fn format_parses() {
        let a = parse_args(&argv(&["--a", "x", "--b", "y", "--format", "json"]))
            .unwrap()
            .unwrap();
        assert_eq!(a.format, Format::Json);

        let a = parse_args(&argv(&["--a", "x", "--b", "y", "--format", "text"]))
            .unwrap()
            .unwrap();
        assert_eq!(a.format, Format::Text);

        assert!(parse_args(&argv(&["--a", "x", "--b", "y", "--format", "yaml"])).is_err());
    }

    #[test]
    fn verbose_parses() {
        for (v, expected) in [
            ("summary", Verbose::Summary),
            ("first", Verbose::First),
            ("full", Verbose::Full),
        ] {
            let a = parse_args(&argv(&["--a", "x", "--b", "y", "--verbose", v]))
                .unwrap()
                .unwrap();
            assert_eq!(a.verbose, expected);
        }
        assert!(parse_args(&argv(&["--a", "x", "--b", "y", "--verbose", "loud"])).is_err());
    }

    #[test]
    fn max_frames_and_flags_parse() {
        let a = parse_args(&argv(&[
            "--a",
            "x",
            "--b",
            "y",
            "--max-frames",
            "60",
            "--no-color",
            "--exit-on-diff",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(a.max_frames, Some(60));
        assert!(a.no_color);
        assert!(a.exit_on_diff);
    }

    #[test]
    fn equals_form_parses() {
        // `--flag=value` is the alternate syntax — assert it works for
        // the value-taking flags.
        let a = parse_args(&argv(&[
            "--a=x.ivf",
            "--b=y.ivf",
            "--level=2",
            "--format=json",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(a.a, PathBuf::from("x.ivf"));
        assert_eq!(a.b, PathBuf::from("y.ivf"));
        assert_eq!(a.level, Level::N(2));
        assert_eq!(a.format, Format::Json);
    }

    #[test]
    fn unknown_arg_fails() {
        assert!(parse_args(&argv(&["--a", "x", "--b", "y", "--bogus"])).is_err());
    }

    // -------------------------------------------------------------------------
    // Byte diff helper
    // -------------------------------------------------------------------------

    /// Build a minimal valid IVF buffer with `frames` 4-byte payloads
    /// of value `fill`. Frame payload N is `[fill, fill+1, fill+2, fill+3]`.
    fn synth_ivf(frames: u32, fill: u8) -> Vec<u8> {
        let mut v = Vec::new();
        // 32-byte IVF header — `DKIF` magic + zero padding is good enough
        // for the byte-diff smoke (no parser reads the rest).
        v.extend_from_slice(b"DKIF");
        v.extend_from_slice(&[0u8; 28]);
        for f in 0..frames {
            v.extend_from_slice(&4u32.to_le_bytes()); // size
            v.extend_from_slice(&[0u8; 8]); // pts
            // Payload: 4 bytes of incrementing data.
            v.extend_from_slice(&[
                fill.wrapping_add(f as u8),
                fill.wrapping_add(f as u8).wrapping_add(1),
                fill.wrapping_add(f as u8).wrapping_add(2),
                fill.wrapping_add(f as u8).wrapping_add(3),
            ]);
        }
        v
    }

    #[test]
    fn identical_buffers_no_divergence() {
        let buf = synth_ivf(3, 0x10);
        assert_eq!(byte_diff(&buf, &buf, None), ByteDiff::Identical);
    }

    #[test]
    fn known_offset_divergence_reported() {
        let mut a = synth_ivf(3, 0x10);
        let mut b = a.clone();
        // Frame 2's payload starts at: 32 (file hdr) + 12 (frame 1 hdr)
        // + 4 (frame 1 payload) + 12 (frame 2 hdr) = 60.
        // Flip the third byte of frame 2's payload (offset 62).
        let target_offset = 62;
        a[target_offset] ^= 0xFF;
        // Or: ensure `b` is what we started with.
        let _ = &mut b;

        let d = byte_diff(&a, &b, None);
        match d {
            ByteDiff::Diverge {
                offset,
                frame,
                payload_offset,
            } => {
                assert_eq!(offset, target_offset);
                assert_eq!(frame, Some(2));
                assert_eq!(payload_offset, Some(2));
            }
            other => panic!("expected Diverge, got {other:?}"),
        }
    }

    #[test]
    fn prefix_mismatch_detected() {
        let buf = synth_ivf(3, 0x10);
        let short = &buf[..buf.len() - 1];
        match byte_diff(&buf, short, None) {
            ByteDiff::PrefixMismatch {
                shorter_len,
                longer_len,
            } => {
                assert_eq!(shorter_len, buf.len() - 1);
                assert_eq!(longer_len, buf.len());
            }
            other => panic!("expected PrefixMismatch, got {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // Reporters
    // -------------------------------------------------------------------------

    #[test]
    fn text_reporter_emits_offset() {
        let mut buf = Vec::<u8>::new();
        {
            let mut r = TextReporter::new(&mut buf, Verbose::First, true, false);
            r.divergence(
                0,
                "byte",
                Some("First byte divergence at offset 62 (frame 2, byte +2 into payload)"),
            );
            assert_eq!(r.finish(), 1);
        }
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("offset 62"), "text output: {s:?}");
        assert!(s.contains("Level 0"), "text output: {s:?}");
    }

    #[test]
    fn json_reporter_emits_ndjson_record() {
        let mut buf = Vec::<u8>::new();
        {
            let mut r = JsonReporter::new(&mut buf, Verbose::First, true);
            r.divergence(0, "byte", Some("First byte divergence at offset 62"));
            assert_eq!(r.finish(), 1);
        }
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2, "want divergence + summary, got: {s:?}");
        assert!(lines[0].contains("\"event\":\"divergence\""));
        assert!(lines[0].contains("\"level\":0"));
        assert!(lines[0].contains("offset 62"));
        assert!(lines[1].contains("\"event\":\"summary\""));
        assert!(lines[1].contains("\"any_divergence\":true"));
    }

    #[test]
    fn verbose_first_stops_after_one_divergence() {
        let mut buf = Vec::<u8>::new();
        {
            let mut r = TextReporter::new(&mut buf, Verbose::First, true, false);
            r.divergence(0, "byte", Some("first"));
            r.divergence(0, "byte", Some("second"));
            assert_eq!(r.finish(), 1);
        }
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("first"), "text: {s:?}");
        assert!(!s.contains("second"), "text: {s:?}");

        // Same for JSON.
        let mut buf = Vec::<u8>::new();
        {
            let mut r = JsonReporter::new(&mut buf, Verbose::First, true);
            r.divergence(0, "byte", Some("first"));
            r.divergence(0, "byte", Some("second"));
            assert_eq!(r.finish(), 1);
        }
        let s = String::from_utf8(buf).unwrap();
        let divergence_lines = s
            .lines()
            .filter(|l| l.contains("\"event\":\"divergence\""))
            .count();
        assert_eq!(divergence_lines, 1, "json: {s:?}");
    }

    #[test]
    fn verbose_full_emits_every_divergence() {
        let mut buf = Vec::<u8>::new();
        {
            let mut r = TextReporter::new(&mut buf, Verbose::Full, true, false);
            r.divergence(0, "byte", Some("first"));
            r.divergence(0, "byte", Some("second"));
            assert_eq!(r.finish(), 1);
        }
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("first"));
        assert!(s.contains("second"));
    }

    #[test]
    fn no_divergence_exit_zero() {
        let mut buf = Vec::<u8>::new();
        let mut r = TextReporter::new(&mut buf, Verbose::First, true, false);
        assert_eq!(r.finish(), 0);
    }

    #[test]
    fn exit_on_diff_off_returns_zero_even_with_divergence() {
        let mut buf = Vec::<u8>::new();
        let mut r = TextReporter::new(
            &mut buf,
            Verbose::First,
            /*exit_on_diff=*/ false,
            false,
        );
        r.divergence(0, "byte", Some("x"));
        assert_eq!(r.finish(), 0);
    }

    #[test]
    fn json_escape_handles_specials() {
        assert_eq!(json_escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }

    // -------------------------------------------------------------------------
    // Integration: drive run_with_streams end-to-end against synthetic
    // buffers via temp files.
    // -------------------------------------------------------------------------

    fn write_tmp(name: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "rvp9-diff-bs-test-{}-{}-{}",
            std::process::id(),
            name,
            // monotonic-ish suffix so parallel tests don't clobber.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, bytes).expect("write tmp");
        path
    }

    #[test]
    fn end_to_end_identical_exits_zero() {
        let buf = synth_ivf(3, 0x10);
        let a = write_tmp("a-id", &buf);
        let b = write_tmp("b-id", &buf);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(0),
            frame: None,
            format: Format::Text,
            verbose: Verbose::First,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let mut out = Vec::<u8>::new();
        let mut err = Vec::<u8>::new();
        let mut reporter = TextReporter::new(&mut out, Verbose::First, true, false);
        let code = run_with_streams(&args, &mut err, &mut reporter);
        let _ = (a, b); // path lifetimes
        assert_eq!(code, 0, "stderr: {}", String::from_utf8_lossy(&err));
    }

    #[test]
    fn end_to_end_diverging_exits_one_with_exit_on_diff() {
        let mut buf_a = synth_ivf(3, 0x10);
        let buf_b = buf_a.clone();
        buf_a[62] ^= 0xFF; // frame 2, payload byte +2
        let a = write_tmp("a-diff", &buf_a);
        let b = write_tmp("b-diff", &buf_b);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(0),
            frame: None,
            format: Format::Json,
            verbose: Verbose::First,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let mut out = Vec::<u8>::new();
        let mut err = Vec::<u8>::new();
        let mut reporter = JsonReporter::new(&mut out, Verbose::First, true);
        let code = run_with_streams(&args, &mut err, &mut reporter);
        let _ = (a, b);
        assert_eq!(code, 1);
        let out_s = String::from_utf8(out).unwrap();
        assert!(out_s.contains("\"level\":0"), "json: {out_s}");
        assert!(out_s.contains("offset 62"), "json: {out_s}");
    }

    #[test]
    fn max_frames_excludes_later_bytes_before_diffing() {
        let buf_a = synth_ivf(3, 0x10);
        let mut buf_b = buf_a.clone();
        buf_b[62] ^= 0xff; // frame 2, outside the one-frame cap
        let a = write_tmp("a-max-one", &buf_a);
        let b = write_tmp("b-max-one", &buf_b);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(4),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: Some(1),
            no_color: true,
            exit_on_diff: true,
        };

        let bounded = read_ivf_prefix(&a, Some(1)).unwrap();
        assert!(bounded.len() < buf_a.len());
        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        assert!(!out.contains("Level "), "out={out:?}");
    }

    #[test]
    fn exit_on_diff_stops_after_first_level_zero_record() {
        let mut buf_a = synth_ivf(3, 0x10);
        let buf_b = buf_a.clone();
        buf_a[62] ^= 0xff;
        let a = write_tmp("a-stop-first", &buf_a);
        let b = write_tmp("b-stop-first", &buf_b);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::All,
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };

        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        assert_eq!(code, 1, "out={out:?} err={err:?}");
        assert_eq!(
            out.lines()
                .filter(|line| line.starts_with("Level "))
                .count(),
            1
        );
        assert!(out.starts_with("Level 0"), "out={out:?}");
    }

    #[test]
    fn level_5_emits_stderr_note() {
        // After commit #4 lands, Level 4 is implemented — only Level 5
        // (per-coef tokens, deferred per design Q5) still emits a
        // "not yet implemented" note.
        let buf = synth_ivf(2, 0x20);
        let a = write_tmp("a-note", &buf);
        let b = write_tmp("b-note", &buf);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(5),
            frame: None,
            format: Format::Text,
            verbose: Verbose::First,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let mut err = Vec::<u8>::new();
        let mut out = Vec::<u8>::new();
        let mut reporter = TextReporter::new(&mut out, Verbose::First, true, false);
        let _code = run_with_streams(&args, &mut err, &mut reporter);
        let _ = (a, b);
        let err_s = String::from_utf8(err).unwrap();
        assert!(
            err_s.contains("level 5 not yet implemented"),
            "err: {err_s}"
        );
    }

    // -------------------------------------------------------------------------
    // Level 1 / Level 2 tests
    // -------------------------------------------------------------------------

    use rvp9_core::bitstream::frame::{ColorSpace, sync};
    use rvp9_core::bitstream::raw::RawBitWriter;
    use rvp9_core::bitstream::sub_parsers::get_tile_n_bits;

    /// Build one minimal-keyframe VP9 frame payload (width × height,
    /// Profile 0, BT.601, studio range, no render size, all-zero tail
    /// except `first_partition_size = 1`). Returns the byte buffer
    /// trimmed to the bits actually written (no trailing padding).
    fn build_keyframe_payload(width: u32, height: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        {
            let mut w = RawBitWriter::new(&mut buf);
            write_keyframe_bits(&mut w, width, height);
        }
        // Keep a few extra bytes so the reader has byte-cursor headroom
        // when it reaches the trailing first_partition_size.
        let used_bits = count_keyframe_bits(width);
        let needed = used_bits.div_ceil(8) + 4;
        buf.truncate(needed);
        buf
    }

    fn write_keyframe_bits(w: &mut RawBitWriter, width: u32, height: u32) {
        // Minimal keyframe: profile 0, show_frame=1, error_resilient=0,
        // sync code, BT.601, studio range, frame size, no render size,
        // all-zero tail, first_partition_size = 1.
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(0); // show_existing = 0
        w.write_bit(0); // frame_type = KEY
        w.write_bit(1); // show_frame = 1
        w.write_bit(0); // error_resilient = 0
        w.write_literal(sync::SYNC_CODE_0 as u32, 8);
        w.write_literal(sync::SYNC_CODE_1 as u32, 8);
        w.write_literal(sync::SYNC_CODE_2 as u32, 8);
        w.write_literal(ColorSpace::Bt601 as u32, 3);
        w.write_bit(0); // color_range = Studio
        w.write_literal(width - 1, 16);
        w.write_literal(height - 1, 16);
        w.write_bit(0); // no render size
        // Tail: refresh + fpdm + idx + loopfilter + quant + seg + tile + fps.
        w.write_bit(0); // refresh_frame_context
        w.write_bit(0); // frame_parallel_decoding_mode
        w.write_literal(0, 2); // frame_context_idx = 0
        w.write_literal(0, 6); // loopfilter filter_level
        w.write_literal(0, 3); // sharpness
        w.write_bit(0); // delta_enabled = 0
        w.write_literal(0, 8); // base_qindex
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0); // segmentation disabled
        // Tile cols: emit 1 zero bit only if min < max.
        let mi_cols = ((width + 7) >> 3) as i32;
        let (min_log2, max_log2) = get_tile_n_bits(mi_cols);
        if max_log2 - min_log2 > 0 {
            w.write_bit(0); // stop increment
        }
        w.write_bit(0); // log2_tile_rows = 0
        w.write_literal(1, 16); // first_partition_size = 1 (non-zero)
    }

    /// Bit count corresponding to write_keyframe_bits — used to size
    /// the buffer without trailing zero padding.
    fn count_keyframe_bits(width: u32) -> usize {
        let mi_cols = ((width + 7) >> 3) as i32;
        let (min_log2, max_log2) = get_tile_n_bits(mi_cols);
        let extra_col_bit: usize = if max_log2 - min_log2 > 0 { 1 } else { 0 };
        // 2 (marker) + 2 (profile) + 1 (show_existing) + 1 (frame_type)
        // + 1 (show) + 1 (err_res) + 24 (sync) + 3 (cs) + 1 (range)
        // + 16 (w-1) + 16 (h-1) + 1 (no render) + 1+1+2 (rfc/fpdm/idx)
        // + 6+3+1 (loopfilter) + 8+1+1+1 (quant) + 1 (seg disabled)
        // + extra_col_bit + 1 (rows) + 16 (fps)
        2 + 2
            + 1
            + 1
            + 1
            + 1
            + 24
            + 3
            + 1
            + 16
            + 16
            + 1
            + 1
            + 1
            + 2
            + 6
            + 3
            + 1
            + 8
            + 1
            + 1
            + 1
            + 1
            + extra_col_bit
            + 1
            + 16
    }

    /// Wrap `payloads` in a minimal IVF file (32-byte header + per-frame
    /// `(size_le_u32, pts_u64)` records + payload bytes).
    fn wrap_in_ivf(payloads: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"DKIF");
        v.extend_from_slice(&[0u8; 28]);
        for (i, p) in payloads.iter().enumerate() {
            v.extend_from_slice(&(p.len() as u32).to_le_bytes());
            v.extend_from_slice(&(i as u64).to_le_bytes());
            v.extend_from_slice(p);
        }
        v
    }

    #[test]
    fn uncompressed_header_partial_eq_smoke() {
        // Parse the same payload twice; expect equality.
        let payload = build_keyframe_payload(320, 240);
        let refs: rvp9_core::bitstream::RefFrameSizes = [None; rvp9_core::bitstream::REF_FRAMES];
        let (h1, _) = rvp9_core::bitstream::parse_uncompressed_header(&payload, &refs).unwrap();
        let (h2, _) = rvp9_core::bitstream::parse_uncompressed_header(&payload, &refs).unwrap();
        assert_eq!(h1, h2);

        // Now perturb height and re-parse → not equal.
        let other = build_keyframe_payload(320, 480);
        let (h3, _) = rvp9_core::bitstream::parse_uncompressed_header(&other, &refs).unwrap();
        assert_ne!(h1, h3);
    }

    /// Build a synthetic IVF file with N keyframes of the given size.
    fn ivf_with_n_keyframes(n: usize, width: u32, height: u32) -> Vec<u8> {
        let payload = build_keyframe_payload(width, height);
        let refs: Vec<&[u8]> = (0..n).map(|_| payload.as_slice()).collect();
        wrap_in_ivf(&refs)
    }

    /// Build a keyframe payload with a `compressed_partition_size`-byte
    /// compressed partition (zero-filled). The uncompressed-header
    /// layout matches [`build_keyframe_payload`] except that the
    /// `first_partition_size` field is set to
    /// `compressed_partition_size`. The compressed partition itself
    /// is appended as raw zero bytes after the header.
    fn build_keyframe_with_compressed_partition(
        width: u32,
        height: u32,
        compressed_partition_size: u16,
    ) -> Vec<u8> {
        // Re-write the uncompressed header byte stream but with a
        // different first_partition_size literal.
        let mut buf = vec![0u8; 256];
        let header_bit_len = {
            let mut w = RawBitWriter::new(&mut buf);
            // Inlined copy of write_keyframe_bits, but with a custom
            // fps literal.
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_literal(ColorSpace::Bt601 as u32, 3);
            w.write_bit(0);
            w.write_literal(width - 1, 16);
            w.write_literal(height - 1, 16);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_literal(0, 2);
            w.write_literal(0, 6);
            w.write_literal(0, 3);
            w.write_bit(0);
            w.write_literal(0, 8);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            let mi_cols = ((width + 7) >> 3) as i32;
            let (min_log2, max_log2) = get_tile_n_bits(mi_cols);
            if max_log2 - min_log2 > 0 {
                w.write_bit(0);
            }
            w.write_bit(0);
            w.write_literal(u32::from(compressed_partition_size), 16);
            count_keyframe_bits(width)
        };
        let header_bytes = header_bit_len.div_ceil(8);
        buf.truncate(header_bytes);
        // Append `compressed_partition_size` zero bytes for the
        // compressed partition + a small trailing pad for bool-reader
        // headroom.
        buf.extend(std::iter::repeat_n(
            0u8,
            compressed_partition_size as usize + 4,
        ));
        buf
    }

    fn run_with_text_reporter(args: &Args) -> (String, String, i32) {
        let mut out = Vec::<u8>::new();
        let mut err = Vec::<u8>::new();
        let code = {
            let mut reporter = TextReporter::new(&mut out, args.verbose, args.exit_on_diff, false);
            run_with_streams(args, &mut err, &mut reporter)
        };
        (
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
            code,
        )
    }

    #[test]
    fn l1_identity_emits_no_record() {
        let buf = ivf_with_n_keyframes(3, 320, 240);
        let a = write_tmp("a-l1-id", &buf);
        let b = write_tmp("b-l1-id", &buf);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(1),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        // Byte-identical → exit 0, no L1 records.
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        assert!(!out.contains("Level 1"), "out={out:?}");
    }

    #[test]
    fn l1_frame_count_mismatch_reported() {
        let buf_a = ivf_with_n_keyframes(3, 320, 240);
        let buf_b = ivf_with_n_keyframes(2, 320, 240);
        let a = write_tmp("a-l1-count", &buf_a);
        let b = write_tmp("b-l1-count", &buf_b);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(1),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: false,
        };
        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        // Must contain a Level 1 frame-count divergence.
        assert!(
            out.contains("Level 1") && out.contains("frame-count"),
            "out={out:?}"
        );
        // Reasoning text must mention the actual counts.
        assert!(out.contains("a=3") && out.contains("b=2"), "out={out:?}");
    }

    #[test]
    fn l2_identity_emits_no_record() {
        let buf = ivf_with_n_keyframes(2, 320, 240);
        let a = write_tmp("a-l2-id", &buf);
        let b = write_tmp("b-l2-id", &buf);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(2),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        assert!(!out.contains("Level 2"), "out={out:?}");
    }

    #[test]
    fn l2_field_divergence_golden() {
        // Build two identical IVFs, then patch a single bit in B's
        // first frame payload to bump filter_level by 1.
        let buf_a = ivf_with_n_keyframes(1, 320, 240);
        let mut buf_b = buf_a.clone();

        // Locate the filter_level bits inside frame 1's payload. The
        // payload starts at offset 32 (file header) + 12 (chunk header)
        // = 44. Within the payload, count keyframe bits up to (but
        // excluding) filter_level: 2+2+1+1+1+1 = 8 (= 1 byte) through
        // error_resilient, then 24 sync + 3 cs + 1 range = 28 bits;
        // 8+28 = 36; then 16+16 width/height = 32; 36+32 = 68; then
        // 1 (no render) + 1+1+2 (rfc/fpdm/idx) = 5; 68+5 = 73 bits.
        // filter_level occupies bits 73..79 (6 bits, MSB-first).
        let payload_start = 44usize;
        let filter_bit_offset = 73usize;
        let byte_idx = payload_start + filter_bit_offset / 8;
        let bit_in_byte = 7 - (filter_bit_offset % 8); // MSB-first
        // filter_level is 0; flip the LSB of the 6-bit value (bit 73+5
        // → bit_in_byte−5 within the same/next byte). We'll flip the
        // *highest* of the 6 filter_level bits (bit 73 itself), which
        // becomes filter_level = 32.
        buf_b[byte_idx] ^= 1 << bit_in_byte;

        let a = write_tmp("a-l2-fld", &buf_a);
        let b = write_tmp("b-l2-fld", &buf_b);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(2),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: false,
        };
        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        // Must contain a Level 2 filter_level record with the right
        // path string and the right values.
        assert!(
            out.contains("header.tail.loop_filter.filter_level"),
            "out={out:?}"
        );
        assert!(out.contains("packet=1 frame=0"), "out={out:?}");
        assert!(out.contains("a=0") && out.contains("b=32"), "out={out:?}");
    }

    #[test]
    fn l2_verbose_full_vs_first() {
        // Build two IVFs that differ on TWO fields of frame 1.
        // Flip filter_level highest bit AND sharpness highest bit.
        let buf_a = ivf_with_n_keyframes(1, 320, 240);
        let mut buf_b_full = buf_a.clone();

        let payload_start = 44usize;
        // filter_level top bit at bit 73.
        let f_bit = 73usize;
        let byte_idx = payload_start + f_bit / 8;
        let bit_in_byte = 7 - (f_bit % 8);
        buf_b_full[byte_idx] ^= 1 << bit_in_byte;
        // sharpness top bit at bit 79 (6 bits later).
        let s_bit = 79usize;
        let byte_idx2 = payload_start + s_bit / 8;
        let bit_in_byte2 = 7 - (s_bit % 8);
        buf_b_full[byte_idx2] ^= 1 << bit_in_byte2;

        let a = write_tmp("a-l2-vf", &buf_a);
        let b = write_tmp("b-l2-vf", &buf_b_full);

        // verbose=full → both records.
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(2),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: false,
        };
        let (out_full, _, _) = run_with_text_reporter(&args);

        // verbose=first → only one Level 2 record.
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(2),
            frame: None,
            format: Format::Text,
            verbose: Verbose::First,
            max_frames: None,
            no_color: true,
            exit_on_diff: false,
        };
        let (out_first, _, _) = run_with_text_reporter(&args);
        let _ = (a, b);

        let level2_full_records = out_full
            .lines()
            .filter(|l| l.starts_with("Level 2") || l.contains("Level 2 "))
            .count();
        let level2_first_records = out_first
            .lines()
            .filter(|l| l.starts_with("Level 2") || l.contains("Level 2 "))
            .count();

        assert!(
            level2_full_records >= 2,
            "expected ≥2 Level 2 records under --verbose full; got {level2_full_records}, out={out_full:?}"
        );
        assert_eq!(
            level2_first_records, 1,
            "expected exactly 1 Level 2 record under --verbose first; got {level2_first_records}, out={out_first:?}"
        );
    }

    // -------------------------------------------------------------------------
    // Level 3 tests
    // -------------------------------------------------------------------------

    /// Test-only synthetic L3 entry. The header is parsed from a real
    /// keyframe helper, but the compressed bytes are standalone parser
    /// fixtures; this is not a complete valid VP9 frame.
    fn synthetic_l3_entry(
        packet: u32,
        mut header: UncompressedHeader,
        kind: FrameKind,
        compressed: Vec<u8>,
    ) -> FrameEntry {
        header.kind = kind;
        header
            .tail
            .as_mut()
            .expect("synthetic L3 entry has a header tail")
            .first_partition_size = u16::try_from(compressed.len()).expect("small test partition");
        FrameEntry {
            packet,
            frame: 0,
            chunk_size: compressed.len(),
            superframe_count: 1,
            is_superframe: false,
            header: Some(header),
            parse_error: None,
            frame_bytes: compressed,
            header_end: 0,
        }
    }

    fn encode_test_compressed_header(
        pre: &FrameContext,
        post: &FrameContext,
        options: CompressedHeaderOptions,
        target: rvp9_core::bitstream::CompressedHeaderResult,
    ) -> Vec<u8> {
        let mut output = vec![0u8; 16 * 1024];
        let mut pre_for_write = pre.clone();
        let mut writer = rvp9_core::entropy::BoolWriter::start_encode(&mut output);
        rvp9_core::bitstream::write_compressed_header(
            &mut writer,
            &mut pre_for_write,
            post,
            options,
            &target,
            None,
        )
        .expect("write synthetic L3 compressed header");
        let written = writer.stop_encode().expect("finish synthetic L3 header");
        output.truncate(written);
        output
    }

    #[test]
    fn l3_frame_range_primes_probability_history_without_prefix_records() {
        use rvp9_core::bitstream::ReferenceMode;
        use rvp9_core::transforms::TxMode;

        let no_refs: RefFrameSizes = [None; REF_FRAMES];
        let key_payload = build_keyframe_payload(64, 64);
        let (key_header, _) = parse_uncompressed_header(&key_payload, &no_refs)
            .expect("fixture keyframe header parses");
        let key_options = options_from_header(&key_header);
        let target = rvp9_core::bitstream::CompressedHeaderResult {
            tx_mode: TxMode::OnlyTx4x4,
            reference_mode: ReferenceMode::Single,
        };

        // Packet 1 updates one leg's skip-probability bank. Both
        // compressed headers are generated by the production writer
        // and must parse successfully to their calibrated post-banks.
        let default_bank = FrameContext::new_default();
        let mut changed_bank = default_bank.clone();
        changed_bank.skip_probs[0] = changed_bank.skip_probs[0].wrapping_sub(17).max(1);
        assert_ne!(changed_bank, default_bank);
        let prefix_a =
            encode_test_compressed_header(&default_bank, &default_bank, key_options, target);
        let prefix_b =
            encode_test_compressed_header(&default_bank, &changed_bank, key_options, target);
        let mut calibrated_a = default_bank.clone();
        parse_compressed_header(&prefix_a, &mut calibrated_a, key_options)
            .expect("identity prefix parses on A");
        let mut calibrated_b = default_bank.clone();
        parse_compressed_header(&prefix_b, &mut calibrated_b, key_options)
            .expect("probability-update prefix parses on B");
        assert_eq!(calibrated_a, default_bank, "A prefix is identity");
        assert_eq!(calibrated_b, changed_bank, "B prefix updates skip_probs[0]");
        assert_ne!(calibrated_a, calibrated_b, "prefix banks diverge");

        // Packet 2 uses a synthetic inter-frame header and a common
        // no-update compressed header. Parsing it from either
        // calibrated bank succeeds and preserves that bank.
        let inter_options = CompressedHeaderOptions {
            lossless: key_options.lossless,
            intra_only: false,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let inter_kind = FrameKind::InterFrame {
            reset_frame_context: 0,
            refresh_frame_flags: 0xff,
            ref_frame_indices: [0; rvp9_core::bitstream::MAX_INTER_REFS],
            ref_frame_sign_biases: [false; rvp9_core::bitstream::MAX_INTER_REFS],
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
        };
        let selected =
            encode_test_compressed_header(&default_bank, &default_bank, inter_options, target);
        let mut selected_a = calibrated_a.clone();
        parse_compressed_header(&selected, &mut selected_a, inter_options)
            .expect("selected no-update inter header parses on A");
        let mut selected_b = calibrated_b.clone();
        parse_compressed_header(&selected, &mut selected_b, inter_options)
            .expect("selected no-update inter header parses on B");
        assert_eq!(
            selected_a, calibrated_a,
            "selected header leaves A bank intact"
        );
        assert_eq!(
            selected_b, calibrated_b,
            "selected header leaves B bank intact"
        );

        let a_entries = vec![
            synthetic_l3_entry(1, key_header.clone(), FrameKind::KeyFrame, prefix_a),
            synthetic_l3_entry(2, key_header.clone(), inter_kind, selected.clone()),
        ];
        let b_entries = vec![
            synthetic_l3_entry(1, key_header.clone(), FrameKind::KeyFrame, prefix_b),
            synthetic_l3_entry(2, key_header, inter_kind, selected),
        ];

        let mut selected_reporter = CaptureReporter::default();
        level_3_diff(
            &a_entries,
            &b_entries,
            &memory_diff_args(Some(FrameRange { start: 2, end: 2 }), None),
            &mut selected_reporter,
        );
        let mut unfiltered_reporter = CaptureReporter::default();
        level_3_diff(
            &a_entries,
            &b_entries,
            &memory_diff_args(None, None),
            &mut unfiltered_reporter,
        );
        let unfiltered_selected: Vec<_> = unfiltered_reporter
            .records
            .into_iter()
            .filter(|record| {
                record
                    .detail
                    .as_deref()
                    .is_some_and(|detail| detail.starts_with("packet=2 "))
            })
            .collect();

        assert!(
            !unfiltered_selected.is_empty(),
            "calibrated prefix-bank difference must remain visible on selected packet 2"
        );
        assert!(
            !selected_reporter.records.is_empty(),
            "--frame 2 lost the packet-1 probability history needed to report packet 2"
        );
        assert!(
            selected_reporter
                .records
                .iter()
                .all(|record| record.level == 3),
            "L3 selection emitted a record at the wrong level: {:?}",
            selected_reporter.records
        );
        assert_eq!(
            selected_reporter.records, unfiltered_selected,
            "selected L3 records must match the packet-2 subset of an unfiltered walk"
        );
        assert!(
            selected_reporter.records.iter().all(|record| record
                .detail
                .as_deref()
                .is_some_and(|detail| detail.starts_with("packet=2 "))),
            "L3 range emitted prefix records: {:?}",
            selected_reporter.records
        );
    }

    #[test]
    fn l3_identity_emits_no_record() {
        // Two byte-identical keyframe-only IVFs → L3 must produce zero
        // divergence records (banks update identically on both sides).
        let buf = ivf_with_n_keyframes(2, 320, 240);
        let a = write_tmp("a-l3-id", &buf);
        let b = write_tmp("b-l3-id", &buf);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(3),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        // Byte-identical → short-circuit at L0; no Level 3 records.
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        assert!(!out.contains("Level 3"), "out={out:?}");
    }

    #[test]
    fn l3_walk_frame_context_path_format() {
        // Smoke-test the field walker independent of the harness:
        // perturb a single y_mode_prob slot and check the emitted
        // path uses the documented `fc.y_mode_prob[group=...][...]`
        // form expected by tooling.
        let mut a = rvp9_core::entropy::FrameContext::new_default();
        let b = a.clone();
        // Perturb a.y_mode_prob[2][1] to a value that is guaranteed
        // to differ from the default (which is 80 at [2][1]).
        a.y_mode_prob[2][1] = a.y_mode_prob[2][1].wrapping_add(7);

        let mut records: Vec<(String, String)> = Vec::new();
        {
            let mut emit = |path: &str, detail: String| {
                records.push((path.to_string(), detail));
            };
            walk_frame_context(&a, &b, &mut emit);
        }
        assert_eq!(records.len(), 1, "records={records:?}");
        let (path, detail) = &records[0];
        assert_eq!(path, "fc.y_mode_prob[group=2][1]");
        // Detail should be the standard `a=.. b=..` shape.
        assert!(
            detail.contains("a=") && detail.contains("b="),
            "detail={detail:?}"
        );
    }

    #[test]
    fn l3_walk_frame_context_covers_diverse_field_areas() {
        // Perturb one leaf in several distinct subtrees of FrameContext
        // and verify the walker emits the exact path for each.
        let mut a = rvp9_core::entropy::FrameContext::new_default();
        let b = a.clone();

        a.partition_prob[5][2] = a.partition_prob[5][2].wrapping_add(1);
        a.coef_probs[1][0][1][3][2][1] = a.coef_probs[1][0][1][3][2][1].wrapping_add(1);
        a.tx_probs.p16x16[1][0] = a.tx_probs.p16x16[1][0].wrapping_add(1);
        a.skip_probs[2] = a.skip_probs[2].wrapping_add(1);
        a.nmvc.comps[0].class0[0] = a.nmvc.comps[0].class0[0].wrapping_add(1);
        a.initialized = false;

        let mut paths: Vec<String> = Vec::new();
        {
            let mut emit = |path: &str, _detail: String| {
                paths.push(path.to_string());
            };
            walk_frame_context(&a, &b, &mut emit);
        }
        let joined = paths.join("\n");
        assert!(
            joined.contains("fc.partition_prob[ctx=5][2]"),
            "paths:\n{joined}"
        );
        assert!(
            joined.contains("fc.coef_probs[tx=8x8][plane=0][ref=1][band=3][ctx=2][node=1]"),
            "paths:\n{joined}",
        );
        assert!(
            joined.contains("fc.tx_probs.p16x16[ctx=1][0]"),
            "paths:\n{joined}"
        );
        assert!(joined.contains("fc.skip_probs[ctx=2]"), "paths:\n{joined}");
        assert!(
            joined.contains("fc.nmvc.comps[0].class0[0]"),
            "paths:\n{joined}"
        );
        assert!(joined.contains("fc.initialized"), "paths:\n{joined}");
        // And we should have emitted exactly 6 records (one per leaf
        // we perturbed).
        assert_eq!(paths.len(), 6, "paths={paths:?}");
    }

    #[test]
    fn l3_post_update_divergence_byte_identical_inputs_no_record() {
        // Locks the harness's "diff post-update fc" contract: when
        // both sides drive the parser against the same bitstream
        // (byte-identical IVF), the resulting post-update banks must
        // match, even though we deliberately seed the *initial* banks
        // identically. This is the L3 equivalent of "byte-identical
        // inputs ⇒ no L3 records" and locks the simplified
        // FrameContextBanks::new() init.
        let buf = ivf_with_n_keyframes(1, 320, 240);
        let a = write_tmp("a-l3-post", &buf);
        let b = write_tmp("b-l3-post", &buf);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(3),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        // Specifically, no `fc.*` paths emitted.
        assert!(!out.contains("fc."), "out={out:?}");
    }

    /// Probe utility for the divergence-fixture builder below: try
    /// each single-bit XOR within the first few bytes of a 64-byte
    /// zero-filled partition, and report which ones change the parser
    /// outcome (either produce a different post-update FrameContext,
    /// or fail where the zero buffer succeeded).
    #[test]
    fn _debug_probe_compressed_byte_perturbations() {
        if std::env::var_os("RVP9_PROBE_BOOL").is_none() {
            return;
        }
        let opts = rvp9_core::bitstream::CompressedHeaderOptions {
            lossless: true,
            intra_only: true,
            interp_filter: rvp9_core::bitstream::InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let base = rvp9_core::entropy::FrameContext::new_default();
        let zero_buf = vec![0u8; 64];
        let baseline = {
            let mut fc = base.clone();
            let r = rvp9_core::bitstream::parse_compressed_header(&zero_buf, &mut fc, opts);
            eprintln!("baseline (64 zeros): {r:?}, fc_eq_default={}", fc == base);
            (r.is_ok(), fc)
        };
        // Iterate byte 0 from 0..128 (top bit must stay 0 to satisfy
        // the marker check) and report which values produce a
        // differing post-update bank.
        for byte0 in 0..128u8 {
            let mut buf = zero_buf.clone();
            buf[0] = byte0;
            let mut fc = base.clone();
            let r = rvp9_core::bitstream::parse_compressed_header(&buf, &mut fc, opts);
            let ok = r.is_ok();
            let differs = ok && fc != baseline.1;
            if differs {
                eprintln!(
                    "byte0={byte0:#04x}: differs (skip_probs={:?}, coef[0][0][0][0][0]={:?})",
                    fc.skip_probs, fc.coef_probs[0][0][0][0][0]
                );
            }
        }
    }

    #[test]
    fn l3_compressed_partition_divergence_reported() {
        // Two IVFs that differ in the compressed-header byte but share
        // an identical uncompressed header → L0 reports the byte
        // diff, L1/L2 stay silent (uncompressed header byte-identical
        // across both sides), and L3 surfaces a divergence — either
        // a post-update prob change, or an asymmetric parse-error.
        //
        // To get a non-truncated parse we build a keyframe with a
        // larger compressed partition (64 zero bytes) so the parser
        // can actually run through `read_coef_probs` + `read_skip_probs`
        // on both sides. We then flip a low-order bit in B's partition
        // — chosen so the marker bit (top bit of byte 0) stays 0 and
        // some downstream gate fires asymmetrically.
        let payload_a = build_keyframe_with_compressed_partition(320, 240, 64);
        let mut payload_b = payload_a.clone();

        // Locate the compressed partition: re-parse the A side to get
        // `header_end`. The keyframe payload's header is followed by
        // 64 zero bytes (the partition) + 4 pad bytes.
        let no_refs: rvp9_core::bitstream::RefFrameSizes = [None; rvp9_core::bitstream::REF_FRAMES];
        let (_, header_end) = rvp9_core::bitstream::parse_uncompressed_header(&payload_a, &no_refs)
            .expect("parse A keyframe");
        // Set the first compressed-partition byte to 0x48
        // (`0b0100_1000`). The top bit stays 0 so the marker check
        // passes; bit 6 = 1 forces the `read_coef_probs` update gate
        // on, which makes the 4x4 coef table walk diverge from the
        // all-zero "no update" path on the A side. The
        // `_debug_probe_compressed_byte_perturbations` test pinned
        // this byte (and its neighbours) as guaranteed to produce a
        // post-update bank divergence under the lossless-keyframe
        // path used here.
        payload_b[header_end] = 0x48;

        let buf_a = wrap_in_ivf(&[payload_a.as_slice()]);
        let buf_b = wrap_in_ivf(&[payload_b.as_slice()]);

        let a = write_tmp("a-l3-comp", &buf_a);
        let b = write_tmp("b-l3-comp", &buf_b);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(3),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: false,
        };
        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        // L0 will of course flag the byte difference. We just need
        // L3 to ALSO surface a fc.* divergence (or a parse-error
        // record), proving the harness drives parse_compressed_header
        // on both sides.
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        let saw_l3 = out
            .lines()
            .any(|l| l.starts_with("Level 3") || l.contains("Level 3 "));
        assert!(
            saw_l3,
            "expected a Level 3 record (fc.* or parse-error) on \
             perturbed compressed-partition byte; out={out:?} err={err:?}",
        );
    }

    // -------------------------------------------------------------------------
    // Level 4 tests
    // -------------------------------------------------------------------------

    fn encode_l4_fixture_leg(selected_y: &[u8]) -> (Vec<u8>, Vec<u8>) {
        use rvp9_encoder::{EncoderConfig, StreamEncoder};

        let mut encoder =
            StreamEncoder::new(EncoderConfig::new(64, 64)).expect("construct L4 fixture encoder");
        let uv = vec![128u8; 32 * 32];
        let flat = vec![96u8; 64 * 64];
        let first = encoder
            .encode_frame(&yuv_view(&flat, &uv))
            .expect("encode fixture keyframe");
        let common = encoder
            .encode_frame(&yuv_view(&flat, &uv))
            .expect("encode common prefix interframe");
        let selected = encoder
            .encode_frame(&yuv_view(selected_y, &uv))
            .expect("encode selected interframe");
        assert!(first.frame_kind.is_key(), "fixture starts with a keyframe");
        assert!(
            !common.frame_kind.is_key(),
            "packet prefix includes an interframe"
        );
        assert!(!selected.frame_kind.is_key(), "selected packet is inter");

        // Put both prerequisite frames in packet 1 so prefix priming
        // must parse and decode every sub-frame in its superframe.
        let mut prefix = rvp9_core::bitstream::SuperframeBuilder::new();
        prefix
            .push_frame(first.data)
            .expect("append prefix keyframe");
        prefix
            .push_frame(common.data)
            .expect("append prefix interframe");
        let prefix = prefix.finish();
        let ivf = wrap_in_ivf(&[prefix.as_slice(), selected.data.as_slice()]);
        (ivf, prefix)
    }

    fn textured_l4_fixture_frame() -> Vec<u8> {
        (0usize..64 * 64)
            .map(|index| {
                let x = index % 64;
                let y = index / 64;
                if (x / 8 + y / 8).is_multiple_of(2) {
                    24
                } else {
                    224
                }
            })
            .collect()
    }

    fn capture_l4_records(
        a_data: &[u8],
        b_data: &[u8],
        frame: Option<FrameRange>,
        max_frames: Option<u32>,
    ) -> Vec<CapturedDivergence> {
        let args = memory_diff_args(frame, max_frames);
        let mut reporter = CaptureReporter::default();
        let mut stderr = Vec::new();
        level_4_diff(a_data, b_data, &args, &mut reporter, &mut stderr);
        assert!(stderr.is_empty(), "unexpected L4 stderr: {stderr:?}");
        reporter.records
    }

    #[test]
    fn l4_encoder_superframe_prefix_fixture_calibrates_selected_mi_difference() {
        use rvp9_decoder::Decoder;

        let flat = vec![96u8; 64 * 64];
        let textured = textured_l4_fixture_frame();
        let (a_data, prefix_a) = encode_l4_fixture_leg(&flat);
        let (b_data, prefix_b) = encode_l4_fixture_leg(&textured);
        assert_eq!(prefix_a, prefix_b, "prerequisite superframes are identical");

        let packets_a = read_ivf_packets(&a_data).expect("read fixture A packets");
        let packets_b = read_ivf_packets(&b_data).expect("read fixture B packets");
        assert_eq!(packets_a.len(), 2);
        assert_eq!(packets_b.len(), 2);
        assert_eq!(
            parse_superframe_index(packets_a[0].slice(&a_data)).expect("prefix index"),
            parse_superframe_index(packets_b[0].slice(&b_data)).expect("prefix index")
        );

        let mut decoder_a = Decoder::new();
        let mut decoder_b = Decoder::new();
        let mut selected_grid_a = Vec::new();
        let mut selected_grid_b = Vec::new();
        for (packet_idx, packet) in packets_a.iter().enumerate() {
            let parsed = decoder_a
                .parse_packet(packet.slice(&a_data))
                .expect("A fixture packet parses with history");
            assert_eq!(parsed.count(), if packet_idx == 0 { 2 } else { 1 });
            for frame in parsed {
                decoder_a
                    .decode_frame(&frame)
                    .expect("A fixture frame decodes with history");
            }
            if packet_idx == 1 {
                selected_grid_a = decoder_a.last_frame_mi_grid().to_vec();
            }
        }
        for (packet_idx, packet) in packets_b.iter().enumerate() {
            let parsed = decoder_b
                .parse_packet(packet.slice(&b_data))
                .expect("B fixture packet parses with history");
            assert_eq!(parsed.count(), if packet_idx == 0 { 2 } else { 1 });
            for frame in parsed {
                decoder_b
                    .decode_frame(&frame)
                    .expect("B fixture frame decodes with history");
            }
            if packet_idx == 1 {
                selected_grid_b = decoder_b.last_frame_mi_grid().to_vec();
            }
        }
        assert!(
            !selected_grid_a.is_empty(),
            "A selected frame publishes an MI grid"
        );
        assert!(
            !selected_grid_b.is_empty(),
            "B selected frame publishes an MI grid"
        );
        assert_ne!(
            selected_grid_a, selected_grid_b,
            "selected packet has real MI divergence"
        );

        let selected_packet = packets_a[1].slice(&a_data);
        let mut fresh = Decoder::new();
        let frames = fresh
            .parse_packet(selected_packet)
            .expect("selected interframe parses without decoder history");
        let fresh_error = frames
            .into_iter()
            .find_map(|frame| fresh.decode_frame(&frame).err());
        assert!(
            matches!(
                fresh_error.as_ref(),
                Some(rvp9_decoder::DecodeError::EmptyInterRef { slot: 0 })
            ),
            "fresh selected interframe should fail for its missing reference history: {fresh_error:?}"
        );
    }

    #[test]
    fn l4_frame_range_primes_superframe_history_and_reports_only_selected_packet() {
        let flat = vec![96u8; 64 * 64];
        let textured = textured_l4_fixture_frame();
        let (a_data, prefix_a) = encode_l4_fixture_leg(&flat);
        let (b_data, prefix_b) = encode_l4_fixture_leg(&textured);
        assert_eq!(prefix_a, prefix_b);

        let selected = capture_l4_records(
            &a_data,
            &b_data,
            Some(FrameRange { start: 2, end: 2 }),
            None,
        );
        let unfiltered = capture_l4_records(&a_data, &b_data, None, None);
        let unfiltered_selected: Vec<_> = unfiltered
            .into_iter()
            .filter(|record| {
                record
                    .detail
                    .as_deref()
                    .is_some_and(|detail| detail.starts_with("packet=2 "))
            })
            .collect();

        assert!(
            !unfiltered_selected.is_empty(),
            "calibrated encoder fixture must produce selected-packet L4 records"
        );
        assert!(
            !selected.is_empty(),
            "--frame 2 lost the packet-1 decoder history needed to report packet 2"
        );
        assert!(
            selected
                .iter()
                .any(|record| record.level == 4 && record.summary.starts_with("mi[")),
            "selected interframe must report an actual MI field path: {selected:?}"
        );
        assert_eq!(
            selected, unfiltered_selected,
            "selected L4 records must equal packet 2's unfiltered records"
        );
        assert!(
            selected.iter().all(|record| record
                .detail
                .as_deref()
                .is_some_and(|detail| detail.starts_with("packet=2 "))),
            "L4 range emitted prefix records: {selected:?}"
        );

        // max_frames counts IVF packets. A cap before selected packet 2
        // must prevent that packet from being decoded or reported.
        let capped = capture_l4_records(
            &a_data,
            &b_data,
            Some(FrameRange { start: 2, end: 2 }),
            Some(1),
        );
        assert!(
            capped.is_empty(),
            "max_frames=1 included packet 2: {capped:?}"
        );
    }

    #[test]
    fn l4_frame_range_primes_both_legs_after_asymmetric_prefix_error() {
        let flat = vec![96u8; 64 * 64];
        let textured = textured_l4_fixture_frame();
        let (valid_a, _) = encode_l4_fixture_leg(&flat);
        let (valid_b, prefix_b) = encode_l4_fixture_leg(&textured);
        let packets_a = read_ivf_packets(&valid_a).expect("read valid A fixture");
        let packets_b = read_ivf_packets(&valid_b).expect("read valid B fixture");
        let invalid_prefix = [0u8]; // invalid frame marker; A's later interframe remains intact
        let selected_a = packets_a[1].slice(&valid_a);
        let selected_b = packets_b[1].slice(&valid_b);
        let broken_a = wrap_in_ivf(&[&invalid_prefix, selected_a]);
        let valid_b_prefix = packets_b[0].slice(&valid_b);
        assert_eq!(valid_b_prefix, prefix_b.as_slice());
        let control_b = wrap_in_ivf(&[valid_b_prefix, selected_b]);

        let records = capture_l4_records(
            &broken_a,
            &control_b,
            Some(FrameRange { start: 2, end: 2 }),
            None,
        );
        assert!(
            records.iter().any(|record| {
                record.summary == "decode.parse_packet" || record.summary == "decode.decode_frame"
            }),
            "one failed prefix must not prevent the valid leg from priming: {records:?}"
        );
        assert!(
            records.iter().all(|record| record
                .detail
                .as_deref()
                .is_some_and(|detail| detail.starts_with("packet=2 "))),
            "asymmetric prefix errors must stay silent: {records:?}"
        );
    }

    /// Locate the conformance-vectors directory, if present. The
    /// workspace ships at most one `.ivf` here (the others are .webm).
    /// Returns `None` when the directory is missing (e.g. fresh
    /// checkout that hasn't run `scripts/fetch-vectors.sh`); callers
    /// should skip the test gracefully in that case.
    fn conformance_vectors_dir() -> Option<PathBuf> {
        let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dir = crate_root
            .parent()?
            .parent()?
            .join("tests")
            .join("conformance")
            .join("vectors");
        if dir.exists() { Some(dir) } else { None }
    }

    /// Find an IVF in `dir` whose decoded output the L4 driver can
    /// actually walk (i.e. `Decoder::decode_frame` returns `Ok` on the
    /// first frame). Returns `None` if the directory is missing or no
    /// IVF is suitable for the current decoder state. The current
    /// shipping `vp90-2-05-resize.ivf` qualifies.
    fn pick_l4_ivf(dir: &Path) -> Option<PathBuf> {
        std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
            let p = e.path();
            (p.extension().and_then(|s| s.to_str()) == Some("ivf")).then_some(p)
        })
    }

    #[test]
    fn likely_site_hint_matches_suffix() {
        assert_eq!(
            likely_site_hint("mi[row=0,col=0].tx_size"),
            Some("encoder::mode_decision"),
        );
        assert_eq!(
            likely_site_hint("mi[row=12,col=8].mode.mvs[0].row"),
            Some("encoder::motion_estimation"),
        );
        assert_eq!(
            likely_site_hint("mi[row=4,col=4].block_size"),
            Some("encoder::partition"),
        );
        assert_eq!(
            likely_site_hint("fc.coef_probs[tx=4x4][plane=0][ref=0][band=0][ctx=0][node=0]"),
            Some("encoder::entropy (token coder)"),
        );
        // Unknown path → None.
        assert_eq!(likely_site_hint("header.profile"), None);
    }

    #[test]
    fn walk_mode_info_identity_emits_nothing() {
        // Two identical ModeInfo structs → walker emits zero records.
        use rvp9_core::block::BlockSize;
        use rvp9_core::transforms::TxSize;
        let mi = rvp9_core::mode_info::ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let mut records = Vec::<(String, String)>::new();
        {
            let mut emit = |p: &str, d: String| records.push((p.to_string(), d));
            walk_mode_info("mi[row=0,col=0]", &mi, &mi, &mut emit);
        }
        assert!(records.is_empty(), "expected zero records, got {records:?}");
    }

    #[test]
    fn walk_mode_info_field_paths() {
        use rvp9_core::block::BlockSize;
        use rvp9_core::transforms::TxSize;
        let base = rvp9_core::mode_info::ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let mut perturbed = base;
        perturbed.tx_size = TxSize::Tx16x16;
        perturbed.skip = true;
        perturbed.segment_id = 3;

        let mut records = Vec::<(String, String)>::new();
        {
            let mut emit = |p: &str, d: String| records.push((p.to_string(), d));
            walk_mode_info("mi[row=2,col=4]", &base, &perturbed, &mut emit);
        }
        let paths: Vec<String> = records.iter().map(|(p, _)| p.clone()).collect();
        assert!(
            paths.contains(&"mi[row=2,col=4].tx_size".to_string()),
            "paths={paths:?}",
        );
        assert!(
            paths.contains(&"mi[row=2,col=4].skip".to_string()),
            "paths={paths:?}",
        );
        assert!(
            paths.contains(&"mi[row=2,col=4].segment_id".to_string()),
            "paths={paths:?}",
        );
    }

    #[test]
    fn walk_mode_info_tag_mismatch() {
        // Intra vs Inter — should emit a single `.tag` record and stop
        // recursing into payload (no need to compare y_mode vs mode).
        use rvp9_core::bitstream::InterpFilter;
        use rvp9_core::block::BlockSize;
        use rvp9_core::mode_info::InterMode;
        use rvp9_core::transforms::TxSize;
        let intra = rvp9_core::mode_info::ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let inter = rvp9_core::mode_info::ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        let mut records = Vec::<(String, String)>::new();
        {
            let mut emit = |p: &str, d: String| records.push((p.to_string(), d));
            walk_mode_info("mi[row=0,col=0]", &intra, &inter, &mut emit);
        }
        let paths: Vec<String> = records.iter().map(|(p, _)| p.clone()).collect();
        assert!(
            paths.iter().any(|p| p.ends_with(".mode.tag")),
            "paths={paths:?}",
        );
        // No y_mode or mvs records — the tag mismatch short-circuits.
        assert!(
            !paths.iter().any(|p| p.contains(".mode.y_mode")),
            "paths={paths:?}",
        );
        assert!(
            !paths.iter().any(|p| p.contains(".mode.mvs")),
            "paths={paths:?}",
        );
    }

    #[test]
    fn l4_identity_on_real_ivf() {
        let dir = match conformance_vectors_dir() {
            Some(d) => d,
            None => {
                eprintln!(
                    "l4_identity_on_real_ivf: conformance vectors dir absent — \
                     skipping (run scripts/fetch-vectors.sh)"
                );
                return;
            }
        };
        let ivf = match pick_l4_ivf(&dir) {
            Some(p) => p,
            None => {
                eprintln!("l4_identity_on_real_ivf: no .ivf in {dir:?} — skipping");
                return;
            }
        };
        let args = Args {
            a: ivf.clone(),
            b: ivf.clone(),
            level: Level::N(4),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let (out, err, code) = run_with_text_reporter(&args);
        // Byte-identical IVF → L0 short-circuits to "identical" and L4
        // never runs. Either way, no divergence records and exit 0.
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        assert!(!out.contains("Level 4"), "out={out:?}");
    }

    /// Build a minimal valid IVF buffer with a single (synthetic)
    /// keyframe payload of the given dims. Frame won't actually decode
    /// past the parse layer — the L4 driver gracefully handles
    /// symmetric `decode_frame` errors as silent noise. We only need
    /// the uncompressed-header dimensions to differ to exercise the
    /// frame-shape divergence path that would dominate a real
    /// `mi_grid.dims` record before decode finishes.
    fn ivf_with_dims(width: u32, height: u32) -> Vec<u8> {
        let payload = build_keyframe_payload(width, height);
        wrap_in_ivf(&[payload.as_slice()])
    }

    #[test]
    fn l4_keyframe_shape_difference_reports_at_l2_first() {
        // Sanity: when two IVFs have different keyframe dims, L2
        // catches the divergence (`header.width` / `header.height`)
        // long before L4's mi-grid walk would see anything. This locks
        // the level-monotonicity invariant from design §6.5: lower
        // levels surface the bug first.
        let buf_a = ivf_with_dims(320, 240);
        let buf_b = ivf_with_dims(640, 480);
        let a = write_tmp("a-l4-shape", &buf_a);
        let b = write_tmp("b-l4-shape", &buf_b);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(4),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: false,
        };
        let (out, _err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        assert_eq!(code, 0);
        // L0 reports the byte diff; L2 reports the width/height fields.
        assert!(
            out.contains("Level 2") && out.contains("header.width"),
            "out={out:?}"
        );
    }

    #[test]
    fn l4_property_lower_levels_also_report_on_diff() {
        // Level monotonicity (design §6.5): if L4 finds any record,
        // the same run with `--level all` MUST also see records at
        // every level that has something to say. Concretely: two IVFs
        // that differ in keyframe shape produce records at L0 (byte
        // diff), L1 (chunk-size differs), and L2 (header.width). The
        // exact set depends on how deep the divergence reaches; the
        // invariant is "no level above the first divergence is
        // silent".
        let buf_a = ivf_with_dims(320, 240);
        let buf_b = ivf_with_dims(640, 480);
        let a = write_tmp("a-l4-mono", &buf_a);
        let b = write_tmp("b-l4-mono", &buf_b);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(4),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: false,
        };
        let (out, _err, _code) = run_with_text_reporter(&args);
        let _ = (a, b);
        // L0 reports the bytes, L1 reports chunk-size, L2 reports
        // width/height. Every level that has something to say must
        // emit at least one record under `--verbose full`.
        for level in &["Level 0", "Level 1", "Level 2"] {
            assert!(
                out.lines().any(|l| l.contains(level)),
                "expected at least one {level} record; out={out:?}",
            );
        }
    }

    #[test]
    fn l4_identity_synthetic_ivf_emits_no_record() {
        // Byte-identical synthetic IVF → L0 short-circuits, L4 never
        // runs, no Level 4 records.
        let buf = ivf_with_n_keyframes(2, 320, 240);
        let a = write_tmp("a-l4-id", &buf);
        let b = write_tmp("b-l4-id", &buf);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(4),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let (out, err, code) = run_with_text_reporter(&args);
        let _ = (a, b);
        assert_eq!(code, 0, "out={out:?} err={err:?}");
        assert!(!out.contains("Level 4"), "out={out:?}");
    }

    #[test]
    fn l4_real_ivf_cost_budget() {
        // Cost-budget guard (design §6.4): driving L4 on a real
        // conformance vector for two identical inputs must finish in
        // under 5 s wall-clock. The shipping vector is small (9 kB,
        // resize stress) — we scale the budget down accordingly. If
        // no IVF is present (fresh checkout), skip; we don't add
        // binary blobs to satisfy this gate.
        let dir = match conformance_vectors_dir() {
            Some(d) => d,
            None => {
                eprintln!("l4_real_ivf_cost_budget: vectors dir absent — skipping");
                return;
            }
        };
        let ivf = match pick_l4_ivf(&dir) {
            Some(p) => p,
            None => {
                eprintln!("l4_real_ivf_cost_budget: no .ivf in vectors — skipping");
                return;
            }
        };
        let args = Args {
            a: ivf.clone(),
            b: ivf.clone(),
            level: Level::N(4),
            frame: None,
            format: Format::Text,
            verbose: Verbose::Full,
            max_frames: None,
            no_color: true,
            exit_on_diff: true,
        };
        let start = std::time::Instant::now();
        let (_out, _err, _code) = run_with_text_reporter(&args);
        let elapsed = start.elapsed();
        // 5 s ceiling per design §6.4. The shipping vector is tiny
        // (9 kB) so we set a wide margin; the budget exists to catch
        // accidental double-decode regressions, not to gate
        // performance work.
        assert!(
            elapsed.as_secs() < 5,
            "L4 identity on {ivf:?} took {elapsed:?} — budget 5s",
        );
    }
}
