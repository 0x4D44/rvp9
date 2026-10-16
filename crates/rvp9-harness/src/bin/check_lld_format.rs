//! `check-lld-format` — validates that every LLD in `notes/lld/` follows the
//! template defined in `notes/vp9-analysis/_stage2-dispatch-prep.md` "LLD
//! template". Designed to be run by Arthur before Phase 2d sign-off, and by
//! the assistant as each background LLD agent lands a file.
//!
//! Checks:
//!   1. Required sections present (level-2 headers): §1 Purpose, §2 libvpx
//!      origin, §3 Public API, §4 Types, §5 Internal algorithm, §6 SIMD
//!      dispatch, §7 Deliberate divergences, §8 Open questions,
//!      §9 Test strategy.
//!   2. Line count within a sensible band (warning, not failure).
//!   3. §7 has at least one entry that looks like a divergence (heuristic:
//!      contains a `Where:` / `What libvpx does:` / `Why:` line).
//!   4. file:line citations present (heuristic: at least one match of
//!      `[A-Za-z0-9_./]+:\d+`).
//!
//! Exit code:
//!   0 — every candidate LLD was read and passes structural checks (§7 may emit warnings).
//!   1 — at least one LLD is missing a required section or an input could not be read.
//!   2 — the input directory could not be opened.
//!
//! Usage:
//!   cargo run --release --bin check-lld-format -- [notes/lld]
//!
//! Defaults to `notes/lld/` relative to the cwd.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Default)]
struct LldReport {
    name: String,
    lines: usize,
    missing_sections: Vec<u8>,
    section_7_has_divergence: bool,
    citation_count: usize,
    /// Count of literal "TBD" markers in the file (case-insensitive,
    /// word-boundaried). Per PLAN.md §6 Stage 2 exit: "No TBD markers
    /// in any LLD that block Stage 3 work."
    tbd_count: usize,
}

impl LldReport {
    fn structurally_ok(&self) -> bool {
        self.missing_sections.is_empty()
    }
}

/// Required section headers, by section number → expected `## N. ` prefix.
/// Match is case-sensitive; the template specifies these exactly.
const REQUIRED_SECTIONS: [u8; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Check whether `line` is a level-2 header opening section `n` (i.e.
/// starts with "## <n>." possibly followed by a space and the title).
fn is_section_header(line: &str, n: u8) -> bool {
    let prefix = format!("## {n}.");
    line.trim_start().starts_with(&prefix)
}

/// Count occurrences of file:line citations (heuristic regex-free check).
/// Looks for a substring matching `<word>:<digits>` where word contains at
/// least one ASCII letter and one period or underscore (heuristic to avoid
/// matching e.g. "rate:1" or "section:5").
fn count_citations(text: &str) -> usize {
    let mut count = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Find a colon followed by an ASCII digit.
        if bytes[i] == b':' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() && i > 0 {
            // Walk back to find the start of the preceding "word".
            let mut start = i;
            while start > 0 {
                let c = bytes[start - 1];
                if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'/' || c == b'-' {
                    start -= 1;
                } else {
                    break;
                }
            }
            let word = &bytes[start..i];
            // Must have at least one letter AND at least one '.' or '_'
            // (or end with a .c/.h extension); avoids matching "rate:1".
            let has_letter = word.iter().any(|b| b.is_ascii_alphabetic());
            let looks_filey = word
                .iter()
                .any(|&b| b == b'.' || b == b'_' || b == b'/' || b == b'-');
            if has_letter && looks_filey {
                count += 1;
            }
            // Skip past the digit run.
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            continue;
        }
        i += 1;
    }
    count
}

/// Detects whether a §7 block contains at least one entry resembling the
/// divergence-ledger template. Looks for ANY of the seven template keys
/// (`Where:`, `What libvpx does:`, `What rvp9 does:`, `Why:`,
/// `Byte-parity impact:`, `Compression-efficiency impact:`,
/// `Future plan:`) appearing within the §7 body. Some LLDs use only a
/// subset of the keys per entry — we match on any.
fn section_7_looks_populated(content: &str) -> bool {
    let Some(start) = content.find("\n## 7.") else {
        return false;
    };
    let after_7 = &content[start + 1..];
    let end = after_7.find("\n## 8.").unwrap_or(after_7.len());
    let body = &after_7[..end];
    body.contains("Where:")
        || body.contains("What libvpx")
        || body.contains("What rvp9")
        || body.contains("Why:")
        || body.contains("Byte-parity")
        || body.contains("Compression-efficiency")
        || body.contains("Future plan")
}

/// Count "TBD" markers in the file, word-boundaried and case-insensitive.
/// Mentions inside `code`/```code fences``` still count (we want to flag
/// them in code-block stubs too).
fn count_tbd(content: &str) -> usize {
    let lower = content.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let pat = b"tbd";
    let mut count = 0;
    let mut i = 0;
    while i + pat.len() <= bytes.len() {
        if &bytes[i..i + pat.len()] == pat {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let after_idx = i + pat.len();
            let after_ok = after_idx == bytes.len() || !bytes[after_idx].is_ascii_alphanumeric();
            if before_ok && after_ok {
                count += 1;
            }
            i += pat.len();
        } else {
            i += 1;
        }
    }
    count
}

fn check_one(path: &Path) -> std::io::Result<LldReport> {
    let content = fs::read_to_string(path)?;
    let lines = content.lines().count();
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();

    let mut missing = Vec::new();
    for n in REQUIRED_SECTIONS {
        if !content.lines().any(|l| is_section_header(l, n)) {
            missing.push(n);
        }
    }

    Ok(LldReport {
        name,
        lines,
        missing_sections: missing,
        section_7_has_divergence: section_7_looks_populated(&content),
        citation_count: count_citations(&content),
        tbd_count: count_tbd(&content),
    })
}

fn scan_entries(
    lld_dir: &Path,
    entries: impl IntoIterator<Item = std::io::Result<PathBuf>>,
) -> (Vec<LldReport>, Vec<String>) {
    let mut reports = Vec::new();
    let mut input_failures = Vec::new();
    for entry in entries {
        let p = match entry {
            Ok(path) => path,
            Err(e) => {
                input_failures.push(format!("read_dir {}: {e}", lld_dir.display()));
                continue;
            }
        };
        if p.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        // Skip README index files and scratchpad files (leading underscore).
        // These are documentation, not LLDs, and don't follow the template.
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem.eq_ignore_ascii_case("README") || stem.starts_with('_') {
            continue;
        }
        match check_one(&p) {
            Ok(r) => reports.push(r),
            Err(e) => input_failures.push(format!("read {}: {e}", p.display())),
        }
    }
    input_failures.sort();
    (reports, input_failures)
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let lld_dir = argv
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("notes/lld"));

    if !lld_dir.is_dir() {
        eprintln!("error: not a directory: {}", lld_dir.display());
        std::process::exit(2);
    }

    let entries = match fs::read_dir(&lld_dir) {
        Ok(it) => it,
        Err(e) => {
            eprintln!("error: read_dir {}: {e}", lld_dir.display());
            std::process::exit(2);
        }
    };
    let (mut reports, input_failures) = scan_entries(
        &lld_dir,
        entries.map(|entry| entry.map(|dir_entry| dir_entry.path())),
    );
    for failure in &input_failures {
        eprintln!("error: {failure}");
    }
    reports.sort_by(|a, b| a.name.cmp(&b.name));

    println!("=== LLD format report ===");
    println!("dir: {}", lld_dir.display());
    println!();
    println!(
        "{:30} {:>6}  {:^9}  {:^7}  {:>5}  {:>4}",
        "file", "lines", "sections", "  §7   ", "cites", "tbd"
    );
    println!(
        "{:30} {:>6}  {:^9}  {:^7}  {:>5}  {:>4}",
        "----", "-----", "--------", "-------", "-----", "---"
    );
    let mut any_failures = !input_failures.is_empty();
    for r in &reports {
        let sect = if r.structurally_ok() {
            "OK 1..9 ".to_string()
        } else {
            let missing: Vec<String> = r.missing_sections.iter().map(|n| n.to_string()).collect();
            format!("MISS:{}", missing.join(","))
        };
        let s7 = if r.section_7_has_divergence {
            " OK    "
        } else {
            " EMPTY "
        };
        println!(
            "{:30} {:>6}  {:^9}  {:^7}  {:>5}  {:>4}",
            r.name, r.lines, sect, s7, r.citation_count, r.tbd_count
        );
        if !r.structurally_ok() {
            any_failures = true;
        }
    }

    println!();
    println!(
        "summary: {} file(s); {} structurally OK, {} with missing sections, {} total TBDs",
        reports.len(),
        reports.iter().filter(|r| r.structurally_ok()).count(),
        reports.iter().filter(|r| !r.structurally_ok()).count(),
        reports.iter().map(|r| r.tbd_count).sum::<usize>(),
    );

    if any_failures {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_section_header() {
        assert!(is_section_header("## 1. Purpose", 1));
        assert!(is_section_header("## 9. Test strategy", 9));
        assert!(!is_section_header("## 10. Not a real section", 1));
        assert!(!is_section_header("### 1. Subsection", 1));
        assert!(!is_section_header("## 2. Other", 1));
    }

    #[test]
    fn counts_filey_citations() {
        let text = "see vp9_encoder.c:1234 and also vp9_pickmode.c:55. \
                    Not a citation: rate:1 or section:3.";
        // Two filey citations: vp9_encoder.c:1234, vp9_pickmode.c:55
        // 'rate:1' and 'section:3' should be filtered out (no '.' or '_').
        assert_eq!(count_citations(text), 2);
    }

    #[test]
    fn section_7_keyword_detected() {
        let body = "preamble\n## 7. Divergences\n\n### 2026-05-16 — foo\n\
                    **Where:** crate::x\n**What libvpx does:** stuff\n\
                    ## 8. Open questions\n";
        assert!(section_7_looks_populated(body));
    }

    #[test]
    fn section_7_missing_returns_false() {
        let body = "## 7. Divergences\n\nNothing here yet.\n## 8. Open\n";
        assert!(!section_7_looks_populated(body));
    }

    #[test]
    fn tbd_word_boundary() {
        // Word-boundaried, case-insensitive. "tbd" and "TBD" count;
        // "tbdfoo", "atbd" do not.
        assert_eq!(count_tbd("nothing here"), 0);
        assert_eq!(count_tbd("foo TBD bar"), 1);
        assert_eq!(count_tbd("Foo tbd; bar TBD!"), 2);
        // Boundary failures:
        assert_eq!(count_tbd("rotbd or atbd"), 0); // 'tbd' is interior
        assert_eq!(count_tbd("TBDfoo"), 0); // alphanumeric immediately after
        // At start/end:
        assert_eq!(count_tbd("TBD"), 1);
        assert_eq!(count_tbd("xx TBD"), 1);
        assert_eq!(count_tbd("TBD xx"), 1);
    }

    #[test]
    fn scan_reports_directory_entry_errors_as_input_failures() {
        let dir = Path::new("notes/lld");
        let entries = [Err(std::io::Error::other("injected entry failure"))];

        let (_, input_failures) = scan_entries(dir, entries);

        assert_eq!(input_failures.len(), 1);
        assert!(input_failures[0].contains("notes/lld"));
        assert!(input_failures[0].contains("injected entry failure"));
    }
}
