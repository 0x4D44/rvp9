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
//! - **skipped** — decoder returned [`DecodeError::Unimplemented`].
//!   Logged + counted, and fails only with `--strict`.
//! - **failed** — packet parsing or decoding failed unexpectedly.
//!   Fails the run unless `--non-gating` was explicitly requested.
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
//!   --official    Run the fail-closed complete-corpus gate: require exactly
//!                 307 vectors and 4,816 displayed-frame matches.
//!   --alpha-supported
//!                 Run the v0.2.0-alpha.1 gate: require all 307 official
//!                 vectors, visibly exclude the one tracked spatial-SVC
//!                 stream, and require all 4,796 supported frames to pass.
//!   --threads N   Construct each decoder with exactly N worker threads.
//!   --verbose     Print one line per frame (default: per-vector
//!                 summary only).
//!   --strict      Treat skipped frames as failures (used by Stage 3
//!                 exit gate once the full decoder is in).
//!   --non-gating  Diagnostic mode: count unexpected parse/decode
//!                 errors as skips instead of failing the vector.

use rvp9_decoder::{DecodeError, Decoder};
use rvp9_harness::conformance_scope::{
    ALPHA_SUPPORTED_FRAME_COUNT, ALPHA_UNSUPPORTED_SPATIAL_SVC_VECTOR, OFFICIAL_FRAME_COUNT,
    OFFICIAL_VECTOR_COUNT, alpha_supported_vectors,
};
use rvp9_harness::container::{ContainerError, read_packets};
use rvp9_harness::md5file::{Md5Entry, Md5FileError, read_md5_file};
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

struct Args {
    vectors_dir: PathBuf,
    limit: Option<usize>,
    official: bool,
    alpha_supported: bool,
    threads: Option<NonZeroUsize>,
    verbose: bool,
    strict: bool,
    non_gating: bool,
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
        official: has("--official"),
        alpha_supported: has("--alpha-supported"),
        threads: has("--threads").then(|| {
            value("--threads")
                .expect("--threads requires a value")
                .parse()
                .ok()
                .and_then(NonZeroUsize::new)
                .expect("--threads must be a positive integer")
        }),
        verbose: has("--verbose"),
        strict: has("--strict"),
        non_gating: has("--non-gating"),
    }
}

fn validate_official_scope(
    official: bool,
    alpha_supported: bool,
    limit: Option<usize>,
    non_gating: bool,
    vector_count: usize,
) -> Result<(), String> {
    if !official {
        return Ok(());
    }
    if alpha_supported {
        return Err("--official cannot be combined with --alpha-supported".to_string());
    }
    if limit.is_some() {
        return Err("--official cannot be combined with --limit".to_string());
    }
    if non_gating {
        return Err("--official cannot be combined with --non-gating".to_string());
    }
    if vector_count != OFFICIAL_VECTOR_COUNT {
        return Err(format!(
            "official mode requires {OFFICIAL_VECTOR_COUNT} vectors, found {vector_count}"
        ));
    }
    Ok(())
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
    non_gating: bool,
    threads: Option<NonZeroUsize>,
    skip_reasons: &mut SkipReasons,
) -> Result<VectorStats, String> {
    let (data, packets) = read_packets(vector_path).map_err(format_container_err)?;
    let md5_entries = read_md5_file(md5_path).map_err(format_md5_err)?;
    let mut stats = VectorStats {
        expected_frames: md5_entries.len(),
        ..VectorStats::default()
    };

    let mut decoder = threads.map_or_else(Decoder::new, Decoder::with_threads);
    let mut shown_idx: usize = 0; // 0-based index into md5_entries

    for (packet_idx, range) in packets.iter().enumerate() {
        let packet_bytes = range.slice(&data);
        let parsed = match decoder.parse_packet(packet_bytes) {
            Ok(p) => p,
            Err(e) => {
                if !non_gating {
                    return Err(format!(
                        "packet {} ({} bytes): parse_packet error: {e}",
                        packet_idx + 1,
                        packet_bytes.len()
                    ));
                }

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

        // Libvpx decodes every superframe component but exposes only the
        // final shown frame from one decode call (`vp9_dx_iface.c:377-415`).
        // Spatial SVC uses one shown component per layer, ordered low to
        // high, so the official ledger contains the final/top layer only.
        let output_frame_index = parsed.output_frame_index();
        let mut displayed = None;
        let mut displayed_skip = None;
        for (frame_index, frame) in parsed.into_iter().enumerate() {
            match decoder.decode_frame(&frame) {
                Ok(buf) => {
                    if Some(frame_index) == output_frame_index && displayed_skip.is_none() {
                        displayed = Some(buf);
                    }
                }
                Err(DecodeError::Unimplemented { reason }) => {
                    displayed = None;
                    displayed_skip.get_or_insert_with(|| format!("unimplemented: {reason}"));
                    break;
                }
                Err(e) => {
                    if !non_gating {
                        return Err(format!(
                            "packet {} at MD5 ledger position {}: decode error: {e}",
                            packet_idx + 1,
                            shown_idx + 1
                        ));
                    }

                    displayed = None;
                    displayed_skip.get_or_insert_with(|| format!("decode error: {e}"));
                    break;
                }
            }
        }

        if let Some(tag) = displayed_skip {
            skip_reasons.record(&tag);
            if output_frame_index.is_some() {
                if verbose {
                    let label = (shown_idx + 1) as u32;
                    println!("    frame {label:>4}: SKIP  {tag}");
                }
                stats.skipped += 1;
                shown_idx += 1;
            }
            continue;
        }
        let Some(buf) = displayed else {
            continue;
        };

        let got_hex = hex16(md5_of_yv12_visible(&buf));
        let entry: Option<&Md5Entry> = md5_entries.get(shown_idx);
        // G8 in-memory ledger no longer carries a 1-based `index` field —
        // derive the displayed-frame index from the row position.
        let display_idx = (shown_idx + 1) as u32;
        match entry {
            Some(e) if e.md5_hex_visible().as_deref() == Some(got_hex.as_str()) => {
                stats.passed += 1;
                if verbose {
                    let (width, height) = if e.width == 0 || e.height == 0 {
                        (buf.width(), buf.height())
                    } else {
                        (e.width, e.height)
                    };
                    println!("    frame {:>4}: PASS  ({}x{})", display_idx, width, height);
                }
            }
            Some(e) => {
                stats.mismatched += 1;
                let want_hex = e
                    .md5_hex_visible()
                    .unwrap_or_else(|| "<no-visible-digest>".to_string());
                println!(
                    "    frame {:>4}: MISMATCH  expected {}  got {}",
                    display_idx, want_hex, got_hex
                );
            }
            None => {
                stats.mismatched += 1;
                println!(
                    "    frame {:>4}: MISMATCH  (no .md5 entry; decoder produced extra frame)",
                    display_idx
                );
            }
        }
        shown_idx += 1;
    }

    if shown_idx != md5_entries.len() {
        return Err(format!(
            "shown-frame count mismatch: decoder accounted for {shown_idx}, MD5 ledger has {}",
            md5_entries.len()
        ));
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
    if let Err(error) = validate_official_scope(
        args.official,
        args.alpha_supported,
        args.limit,
        args.non_gating,
        vectors.len(),
    ) {
        eprintln!("{error}");
        std::process::exit(2);
    }
    if args.alpha_supported && args.limit.is_some() {
        eprintln!("--alpha-supported cannot be combined with --limit");
        std::process::exit(2);
    }
    if args.alpha_supported && args.non_gating {
        eprintln!("--alpha-supported cannot be combined with --non-gating");
        std::process::exit(2);
    }
    if args.alpha_supported {
        vectors = match alpha_supported_vectors(vectors) {
            Ok(selected) => selected,
            Err(error) => {
                eprintln!("vector selection failed: {error}");
                std::process::exit(2);
            }
        };
    }
    if let Some(n) = args.limit {
        vectors.truncate(n);
    }

    if vectors.is_empty() {
        println!(
            "no .ivf/.webm vectors found in {}",
            args.vectors_dir.display()
        );
        println!("(run scripts/fetch-vectors.sh to populate)");
        std::process::exit(0);
    }

    println!(
        "running rvp9 decoder against {} vector(s) in {}",
        vectors.len(),
        args.vectors_dir.display()
    );
    if args.alpha_supported {
        println!(
            "alpha-supported scope: excluded 1/{OFFICIAL_VECTOR_COUNT}: {ALPHA_UNSUPPORTED_SPATIAL_SVC_VECTOR}"
        );
    }
    if args.official {
        println!(
            "official scope: {OFFICIAL_VECTOR_COUNT} vectors / {OFFICIAL_FRAME_COUNT} displayed frames"
        );
    }
    if let Some(threads) = args.threads {
        println!("decoder threads: {threads}");
    }
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
        match run_one_vector(
            vector,
            &md5_path,
            args.verbose,
            args.non_gating,
            args.threads,
            &mut skip_reasons,
        ) {
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
        println!("Vector-level errors: {vector_failures}");
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
    // - Container, ledger, and gating-mode decode failures are hard
    //   failures.
    // - `Unimplemented` skips only fail in `--strict` mode.
    // - Unexpected parser/decoder errors reach this point as skips
    //   only when explicit `--non-gating` diagnostic mode is active.
    let strict_skip_fail =
        (args.strict || args.official || args.alpha_supported) && totals.skipped > 0;
    let expected_passed = if args.official {
        Some(OFFICIAL_FRAME_COUNT)
    } else if args.alpha_supported {
        Some(ALPHA_SUPPORTED_FRAME_COUNT)
    } else {
        None
    };
    let expected_passed_fail = expected_passed.is_some_and(|expected| totals.passed != expected);
    if expected_passed_fail {
        eprintln!(
            "passed-frame gate failed: expected {}, observed {}",
            expected_passed.expect("a failed expected-pass count has an expected value"),
            totals.passed
        );
    }
    if totals.mismatched > 0 || vector_failures > 0 || strict_skip_fail || expected_passed_fail {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_harness::container::{IvfHeader, write_ivf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        vector: PathBuf,
        ledger: PathBuf,
    }

    impl Fixture {
        fn with_packets(packets: &[&[u8]]) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos();
            let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let stem = format!("rvp9-conformance-{}-{nonce}-{sequence}", std::process::id());
            let vector = std::env::temp_dir().join(format!("{stem}.ivf"));
            let ledger = std::env::temp_dir().join(format!("{stem}.ivf.md5"));

            let mut ivf = Vec::new();
            write_ivf(&mut ivf, &IvfHeader::vp9(64, 64, 30, 1), packets)
                .expect("write IVF fixture");
            std::fs::write(&vector, ivf).expect("persist IVF fixture");
            std::fs::write(
                &ledger,
                "00000000000000000000000000000000  malformed-64x64-0001.i420\n",
            )
            .expect("persist malformed IVF ledger");

            Self { vector, ledger }
        }

        fn malformed_packet() -> Self {
            Self::with_packets(&[&[]])
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.vector);
            let _ = std::fs::remove_file(&self.ledger);
        }
    }

    #[test]
    fn malformed_packet_is_fatal_by_default() {
        let fixture = Fixture::malformed_packet();
        let mut skip_reasons = SkipReasons::default();

        let result = run_one_vector(
            &fixture.vector,
            &fixture.ledger,
            false,
            false,
            None,
            &mut skip_reasons,
        );

        assert!(result.is_err(), "malformed packet was silently skipped");
    }

    #[test]
    fn non_gating_mode_can_skip_a_malformed_packet() {
        let fixture = Fixture::malformed_packet();
        let mut skip_reasons = SkipReasons::default();

        let stats = run_one_vector(
            &fixture.vector,
            &fixture.ledger,
            false,
            true,
            None,
            &mut skip_reasons,
        )
        .expect("explicit non-gating mode should preserve diagnostic skips");

        assert_eq!(stats.skipped, 1);
    }

    #[test]
    fn missing_shown_frame_is_fatal_even_in_non_gating_mode() {
        let fixture = Fixture::with_packets(&[]);
        let mut skip_reasons = SkipReasons::default();
        let (_, packets) = read_packets(&fixture.vector).expect("read empty IVF fixture");
        assert!(packets.is_empty());
        let entries = read_md5_file(&fixture.ledger).expect("read fixture ledger");
        assert_eq!(entries.len(), 1);

        let result = run_one_vector(
            &fixture.vector,
            &fixture.ledger,
            false,
            true,
            None,
            &mut skip_reasons,
        );

        let error = match result {
            Ok(_) => panic!("missing shown-frame accounting was not rejected"),
            Err(error) => error,
        };
        assert!(
            error.contains("shown-frame count mismatch"),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn official_scope_is_fail_closed() {
        assert!(validate_official_scope(false, false, Some(1), true, 0).is_ok());
        assert!(validate_official_scope(true, false, None, false, OFFICIAL_VECTOR_COUNT).is_ok());
        assert_eq!(
            validate_official_scope(true, true, None, false, OFFICIAL_VECTOR_COUNT).unwrap_err(),
            "--official cannot be combined with --alpha-supported"
        );
        assert_eq!(
            validate_official_scope(true, false, Some(1), false, OFFICIAL_VECTOR_COUNT)
                .unwrap_err(),
            "--official cannot be combined with --limit"
        );
        assert_eq!(
            validate_official_scope(true, false, None, true, OFFICIAL_VECTOR_COUNT).unwrap_err(),
            "--official cannot be combined with --non-gating"
        );
        assert_eq!(
            validate_official_scope(true, false, None, false, OFFICIAL_VECTOR_COUNT - 1)
                .unwrap_err(),
            "official mode requires 307 vectors, found 306"
        );
    }
}
