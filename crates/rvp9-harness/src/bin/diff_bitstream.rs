//! Bitstream differential diff tool — Level 0 (byte diff).
//!
//! Compares two `.ivf` files and reports the first point of divergence.
//! This binary is the **Level 0** entry point for the structured diff
//! harness described in
//! `notes/stage4-design/diff-bitstream-harness.md`. Higher levels
//! (1..=4) are planned for commits 2-4 of that design and currently
//! emit a "not yet implemented" stderr note while still running the
//! Level 0 byte diff.
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

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

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
            let n: u32 = v
                .parse()
                .map_err(|e| format!("--frame not a u32: {e}"))?;
            Ok(FrameRange { start: n, end: n })
        }
        Some((s, e)) => {
            let start: u32 = s
                .parse()
                .map_err(|e| format!("--frame range start: {e}"))?;
            let end: u32 = e
                .parse()
                .map_err(|e| format!("--frame range end: {e}"))?;
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
                                 (levels 1..=5 not yet implemented)
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
}

/// Text reporter — preserves the original Stage 0 stdout when run at
/// `--level 0` with no extra flags.
struct TextReporter<W: Write> {
    out: W,
    /// Stop after first divergence?
    stop_at_first: bool,
    /// Has any divergence been seen?
    any_diff: bool,
    /// Number of divergences emitted so far (for `--verbose first` gating).
    emitted: usize,
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
            exit_on_diff,
            color,
        }
    }
}

impl<W: Write> DiffReporter for TextReporter<W> {
    fn divergence(&mut self, level: u8, summary: &str, detail: Option<&str>) {
        if self.stop_at_first && self.emitted > 0 {
            self.any_diff = true; // still record that we saw one
            return;
        }
        self.any_diff = true;
        self.emitted += 1;
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
        if self.any_diff && self.exit_on_diff { 1 } else { 0 }
    }
}

/// JSON reporter — one NDJSON record per `divergence()` call, plus a
/// trailing summary object. Format spec: §5.4 of the design.
struct JsonReporter<W: Write> {
    out: W,
    stop_at_first: bool,
    any_diff: bool,
    emitted: usize,
    exit_on_diff: bool,
}

impl<W: Write> JsonReporter<W> {
    fn new(out: W, verbose: Verbose, exit_on_diff: bool) -> Self {
        Self {
            out,
            stop_at_first: !matches!(verbose, Verbose::Full),
            any_diff: false,
            emitted: 0,
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
        if self.stop_at_first && self.emitted > 0 {
            self.any_diff = true;
            return;
        }
        self.any_diff = true;
        self.emitted += 1;
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
        if self.any_diff && self.exit_on_diff { 1 } else { 0 }
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
    PrefixMismatch { shorter_len: usize, longer_len: usize },
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
        } => format!(
            "One is a prefix of the other (lengths differ: {shorter_len} vs {longer_len})."
        ),
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
    // Notes for flags whose behaviour ships in later commits.
    let level_n = args.level.as_u8();
    if level_n >= 1 {
        let _ = writeln!(
            stderr,
            "note: level {level_n} not yet implemented \
             (commits 2-4 of stage4-design/diff-bitstream-harness.md)"
        );
    }
    if args.frame.is_some() {
        let _ = writeln!(
            stderr,
            "note: --frame is ignored at Level 0 (whole-file byte diff)"
        );
    }
    if matches!(args.verbose, Verbose::Full) {
        let _ = writeln!(
            stderr,
            "note: --verbose full not yet implemented for Level 0 \
             (defaults to first-divergence behaviour)"
        );
    }

    let a = match std::fs::read(&args.a) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(stderr, "could not read --a {}: {e}", args.a.display());
            return 2;
        }
    };
    let b = match std::fs::read(&args.b) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(stderr, "could not read --b {}: {e}", args.b.display());
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

    let diff = byte_diff(&a, &b, args.max_frames);
    match &diff {
        ByteDiff::Identical => {
            // No divergence — reporter records nothing. Text format
            // callers surface the "identical" line via the preamble
            // path in main().
        }
        ByteDiff::PrefixMismatch { .. } | ByteDiff::Diverge { .. } => {
            let msg = format_byte_diff(&diff);
            reporter.divergence(0, "byte", Some(&msg));
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

    let code = match args.format {
        Format::Text => {
            let mut reporter = TextReporter::new(&mut out, args.verbose, args.exit_on_diff, color);
            run_with_streams(&args, &mut err, &mut reporter)
        }
        Format::Json => {
            let mut reporter = JsonReporter::new(&mut out, args.verbose, args.exit_on_diff);
            run_with_streams(&args, &mut err, &mut reporter)
        }
    };

    // Original Stage 0 contract: when text format and no divergence,
    // print "Identical: every byte matches." on stdout. The reporter
    // doesn't see "no divergence" events so we surface it here.
    if matches!(args.format, Format::Text) && code == 0 {
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
            let a = parse_args(&argv(&[
                "--a",
                "x",
                "--b",
                "y",
                "--level",
                &n.to_string(),
            ]))
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
        let a = parse_args(&argv(&["--a=x.ivf", "--b=y.ivf", "--level=2", "--format=json"]))
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
        let mut r = TextReporter::new(&mut buf, Verbose::First, /*exit_on_diff=*/ false, false);
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
    fn level_above_zero_emits_stderr_note() {
        let buf = synth_ivf(2, 0x20);
        let a = write_tmp("a-note", &buf);
        let b = write_tmp("b-note", &buf);
        let args = Args {
            a: a.clone(),
            b: b.clone(),
            level: Level::N(4),
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
        assert!(err_s.contains("level 4 not yet implemented"), "err: {err_s}");
    }
}
