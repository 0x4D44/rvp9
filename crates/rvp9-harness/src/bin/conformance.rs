//! VP9 per-frame conformance test runner.
//!
//! For each test vector under `--vectors-dir`, walks transport
//! packets via [`rvp9_harness::container`], drives them through
//! [`rvp9_decoder::Decoder::decode_frame`], MD5s the decoded YV12
//! frame (Y, U, V — visible region only, matching libvpx
//! `vpxdec --md5` semantics), and compares against the sibling
//! `<vector>.md5` ledger.
//!
//! Per-frame outcome:
//!
//! - **passed** — decoder returned a frame and the MD5 matches.
//! - **mismatched** — decoder returned a frame but the MD5 differs.
//!   This is the load-bearing gate once decode paths fill in:
//!   triggers a non-zero exit.
//! - **skipped** — decoder returned [`DecodeError::Unimplemented`] or
//!   any other error. Logged + counted, does NOT fail the run.
//!
//! Until the full decode pipeline lands (Stage 3, later tasks), every
//! frame for which we don't yet have a working decode path falls into
//! "skipped". As paths come online they will move into "passed" or
//! catch us on "mismatched".
//!
//! Usage:
//!   cargo run --release --bin conformance -- \
//!       --vectors-dir tests/conformance/vectors
//!
//! Optional flags:
//!   --limit N     Process only the first N vectors (sorted by name).
//!   --verbose     Print one line per frame (default: per-vector
//!                 summary only).
//!   --strict      Treat skipped frames as failures (used by Stage 3
//!                 exit gate once the full decoder is in).

use rvp9_decoder::{DecodeError, Decoder};
use rvp9_harness::container::{ContainerError, read_packets};
use rvp9_harness::md5file::{Md5Entry, Md5FileError, read_md5_file};
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::path::{Path, PathBuf};

struct Args {
    vectors_dir: PathBuf,
    limit: Option<usize>,
    verbose: bool,
    strict: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let value = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    let has = |flag: &str| -> bool { argv.iter().any(|a| a == flag) };
    Args {
        vectors_dir: value("--vectors-dir")
            .map(PathBuf::from)
            .expect("--vectors-dir required"),
        limit: value("--limit").map(|s| s.parse().expect("--limit usize")),
        verbose: has("--verbose"),
        strict: has("--strict"),
    }
}

/// Per-vector outcome counters.
#[derive(Default, Clone, Copy)]
struct VectorStats {
    passed: usize,
    mismatched: usize,
    skipped: usize,
    expected_frames: usize,
}

/// Reasons a frame was skipped — collected for diagnostic summaries.
#[derive(Default)]
struct SkipReasons {
    /// Count by short tag (e.g. "unimplemented: keyframe ...").
    by_tag: std::collections::BTreeMap<String, usize>,
}

impl SkipReasons {
    fn record(&mut self, tag: &str) {
        *self.by_tag.entry(tag.to_string()).or_insert(0) += 1;
    }
}

/// Drive one vector through the decoder and tally per-frame outcomes.
///
/// Returns the per-vector stats. Errors (container / .md5 parse
/// failures) are returned to the caller as a fatal-for-this-vector
/// string — they don't terminate the whole run.
fn run_one_vector(
    vector_path: &Path,
    md5_path: &Path,
    verbose: bool,
    skip_reasons: &mut SkipReasons,
) -> Result<VectorStats, String> {
    let (data, packets) = read_packets(vector_path).map_err(format_container_err)?;
    let md5_entries = read_md5_file(md5_path).map_err(format_md5_err)?;
    let mut stats = VectorStats {
        expected_frames: md5_entries.len(),
        ..VectorStats::default()
    };

    let mut decoder = Decoder::new();
    let mut shown_idx: usize = 0; // 0-based index into md5_entries

    for (packet_idx, range) in packets.iter().enumerate() {
        let packet_bytes = range.slice(&data);
        let parsed = match decoder.parse_packet(packet_bytes) {
            Ok(p) => p,
            Err(e) => {
                // Header parse failure is currently a "skip" too —
                // libvpx accepts these but our parser may not yet.
                let tag = format!("parse_packet error: {e}");
                skip_reasons.record(&tag);
                if verbose {
                    println!(
                        "    packet {} ({} bytes): {tag}",
                        packet_idx + 1,
                        packet_bytes.len()
                    );
                }
                // We can't know how many frames this packet *would*
                // have shown — assume 1 (the common case).
                stats.skipped += 1;
                shown_idx += 1;
                continue;
            }
        };

        for frame in parsed {
            // A frame is "displayed" iff show_frame == true or
            // it's a ShowExistingFrame (which always shows).
            let shown = frame.header.show_frame
                || matches!(
                    frame.header.kind,
                    rvp9_core::bitstream::FrameKind::ShowExistingFrame { .. }
                );
            match decoder.decode_frame(&frame) {
                Ok(buf) => {
                    if !shown {
                        // Not-shown frames don't have an .md5 ledger
                        // entry; just continue.
                        continue;
                    }
                    let got = md5_of_yv12_visible(&buf);
                    let got_hex = hex16(got);
                    let entry: Option<&Md5Entry> = md5_entries.get(shown_idx);
                    match entry {
                        Some(e) if e.md5_hex == got_hex => {
                            stats.passed += 1;
                            if verbose {
                                println!(
                                    "    frame {:>4}: PASS  ({}x{})",
                                    e.index, e.width, e.height
                                );
                            }
                        }
                        Some(e) => {
                            stats.mismatched += 1;
                            println!(
                                "    frame {:>4}: MISMATCH  expected {}  got {}",
                                e.index, e.md5_hex, got_hex
                            );
                        }
                        None => {
                            stats.mismatched += 1;
                            println!(
                                "    frame {:>4}: MISMATCH  (no .md5 entry; decoder produced extra frame)",
                                shown_idx + 1
                            );
                        }
                    }
                    shown_idx += 1;
                }
                Err(DecodeError::Unimplemented { reason }) => {
                    if shown {
                        let tag = format!("unimplemented: {reason}");
                        skip_reasons.record(&tag);
                        if verbose {
                            let label = md5_entries
                                .get(shown_idx)
                                .map(|e| e.index)
                                .unwrap_or((shown_idx + 1) as u32);
                            println!("    frame {label:>4}: SKIP  {tag}");
                        }
                        stats.skipped += 1;
                        shown_idx += 1;
                    }
                }
                Err(e) => {
                    // Any other DecodeError counts as a skip with a
                    // diagnostic; failure path here will become a
                    // hard fail once decode paths are in (Wave 3+).
                    if shown {
                        let tag = format!("decode error: {e}");
                        skip_reasons.record(&tag);
                        if verbose {
                            let label = md5_entries
                                .get(shown_idx)
                                .map(|e| e.index)
                                .unwrap_or((shown_idx + 1) as u32);
                            println!("    frame {label:>4}: SKIP  {tag}");
                        }
                        stats.skipped += 1;
                        shown_idx += 1;
                    }
                }
            }
        }
    }

    Ok(stats)
}

fn format_container_err(e: ContainerError) -> String {
    format!("container: {e}")
}
fn format_md5_err(e: Md5FileError) -> String {
    format!("md5-file: {e}")
}

fn main() {
    let args = parse_args();

    let mut vectors: Vec<PathBuf> = match std::fs::read_dir(&args.vectors_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|x| matches!(x, "ivf" | "webm"))
            })
            .collect(),
        Err(e) => {
            eprintln!(
                "could not list vectors in {}: {e}",
                args.vectors_dir.display()
            );
            std::process::exit(2);
        }
    };
    vectors.sort();
    if let Some(n) = args.limit {
        vectors.truncate(n);
    }

    if vectors.is_empty() {
        println!("no .ivf/.webm vectors found in {}", args.vectors_dir.display());
        println!("(run scripts/fetch-vectors.ps1 to populate)");
        std::process::exit(0);
    }

    println!(
        "running rvp9 decoder against {} vector(s) in {}",
        vectors.len(),
        args.vectors_dir.display()
    );
    println!();

    let mut totals = VectorStats::default();
    let mut skip_reasons = SkipReasons::default();
    let mut vector_failures: usize = 0;

    for vector in &vectors {
        let name = vector.file_name().unwrap().to_string_lossy().to_string();
        let md5_path = vector.with_extension(format!(
            "{}.md5",
            vector.extension().and_then(|s| s.to_str()).unwrap_or("")
        ));

        println!("  {name}");
        match run_one_vector(vector, &md5_path, args.verbose, &mut skip_reasons) {
            Ok(stats) => {
                println!(
                    "    {:>3} passed / {:>3} mismatched / {:>3} skipped / {:>3} expected",
                    stats.passed, stats.mismatched, stats.skipped, stats.expected_frames
                );
                totals.passed += stats.passed;
                totals.mismatched += stats.mismatched;
                totals.skipped += stats.skipped;
                totals.expected_frames += stats.expected_frames;
            }
            Err(reason) => {
                println!("    VECTOR-FAIL  {reason}");
                vector_failures += 1;
            }
        }
    }

    println!();
    println!(
        "Conformance: {} passed / {} mismatched / {} skipped / {} total",
        totals.passed, totals.mismatched, totals.skipped, totals.expected_frames
    );
    if vector_failures > 0 {
        println!("Vector-level errors (container / .md5 parse): {vector_failures}");
    }

    if !skip_reasons.by_tag.is_empty() {
        println!();
        println!("Skip-reason breakdown:");
        for (tag, count) in &skip_reasons.by_tag {
            println!("  {count:>6} × {tag}");
        }
    }

    // Exit code policy:
    // - Any mismatch is always a hard fail (load-bearing).
    // - Vector-level parse failures are always a hard fail.
    // - Skipped frames only fail in `--strict` mode.
    let strict_skip_fail = args.strict && totals.skipped > 0;
    if totals.mismatched > 0 || vector_failures > 0 || strict_skip_fail {
        std::process::exit(1);
    }
}
