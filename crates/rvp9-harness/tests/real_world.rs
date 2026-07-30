//! G8 — Real-world corpus harness (HLD V6 §G8 lines 1538-1546).
//!
//! Scans `tests/corpus/fixtures/` at test time for VP9 `.ivf` / `.webm`
//! files, decodes each via `rvp9_decoder::Decoder`, MD5s every shown
//! frame, and compares against the sibling `<clip>.md5` reference
//! ledger. Per-clip statistics (total frames decoded, `show_frame=0`
//! count, Profile 0/1 breakdown) are emitted to stdout.
//!
//! ## Skip behaviour
//!
//! When `tests/corpus/fixtures/` is empty (the current G8.0 state —
//! `vpxenc` is not on PATH in this sandbox so G8.1 has not landed),
//! the test passes after printing a single skip line. Once fixtures
//! land via the G8.1 follow-up (`scripts/fetch-corpus.ps1` +
//! `scripts/encode-corpus.ps1`), the test starts running automatically
//! — no harness code change required.
//!
//! ## Why no env var / feature flag?
//!
//! Per CLAUDE.md project policy (no feature flags) and the supervisor
//! decision documented in `notes/g8-followups.md`. The skip-clean path
//! makes the test cheap to keep enabled at all times.

use rvp9_core::bitstream::Profile;
use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use rvp9_harness::md5file::read_md5_file;
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    // `CARGO_MANIFEST_DIR` = `crates/rvp9-harness`; workspace root is
    // two levels up.
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("tests")
        .join("corpus")
        .join("fixtures")
}

/// Per-clip telemetry counter. Mirrored to stdout per the HLD §G8
/// "Per-corpus-clip enumeration" requirement (lines 1543-1546).
#[derive(Default, Debug)]
struct ClipStats {
    total_decoded: usize,
    show_frame_zero: usize,
    profile_0: usize,
    profile_1: usize,
    profile_other: usize,
    visible_md5_mismatched: usize,
    visible_md5_matched: usize,
}

#[test]
fn real_world_corpus_decodes_bit_exact() {
    // G8.0 review (2026-05-21): the `!dir.exists()` early-return arm
    // was unreachable because `tests/corpus/fixtures/README.md` is
    // committed (the directory always exists). Only the empty-after-
    // filter check below can fire — keep that one.
    let dir = fixtures_dir();

    let mut fixtures: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|x| matches!(x, "ivf" | "webm"))
        })
        .collect();
    fixtures.sort();

    if fixtures.is_empty() {
        println!(
            "real_world.rs: no in-repo fixtures present; skipping. \
             Acquire via scripts/fetch-corpus.ps1 + \
             scripts/encode-corpus.ps1 (G8.1)."
        );
        return;
    }

    println!(
        "real_world.rs: walking {} fixture(s) in {}",
        fixtures.len(),
        dir.display()
    );

    let mut total_mismatched = 0usize;
    let mut aggregate_show_frame_zero = 0usize;

    for fixture in &fixtures {
        let md5_path = fixture.with_extension(format!(
            "{}.md5",
            fixture.extension().and_then(|s| s.to_str()).unwrap_or("")
        ));
        let (data, packets) = read_packets(fixture)
            .unwrap_or_else(|e| panic!("container parse {}: {e}", fixture.display()));

        let md5_entries = read_md5_file(&md5_path)
            .unwrap_or_else(|e| panic!("MD5 ledger {}: {e}", md5_path.display()));

        let mut stats = ClipStats::default();
        // Pin to one worker (G5 review M2 — avoid thread amplification).
        let mut decoder = Decoder::with_threads(NonZeroUsize::new(1).expect("1 is non-zero"));
        let mut shown_idx: usize = 0;
        for range in &packets {
            let parsed = decoder
                .parse_packet(range.slice(&data))
                .unwrap_or_else(|e| panic!("parse_packet {}: {e}", fixture.display()));
            for frame in parsed {
                let shown = frame.header.show_frame
                    || matches!(
                        frame.header.kind,
                        rvp9_core::bitstream::FrameKind::ShowExistingFrame { .. }
                    );
                match decoder.decode_frame(&frame) {
                    Ok(buf) => {
                        stats.total_decoded += 1;
                        match frame.header.profile {
                            Profile::Profile0 => stats.profile_0 += 1,
                            Profile::Profile1 => stats.profile_1 += 1,
                            _ => stats.profile_other += 1,
                        }
                        if !shown {
                            stats.show_frame_zero += 1;
                            continue;
                        }
                        let got_hex = hex16(md5_of_yv12_visible(&buf));
                        let want = md5_entries.get(shown_idx);
                        match want {
                            Some(e) if e.md5_hex_visible().as_deref() == Some(got_hex.as_str()) => {
                                stats.visible_md5_matched += 1;
                            }
                            _ => {
                                stats.visible_md5_mismatched += 1;
                            }
                        }
                        shown_idx += 1;
                    }
                    Err(e) => panic!(
                        "{}: decode error on frame {shown_idx} (kind={:?}): {e}",
                        fixture.display(),
                        frame.header.kind
                    ),
                }
            }
        }

        println!(
            "  {}: {} frames decoded ({} P0 / {} P1 / {} other), \
             show_frame=0 count = {}, visible MD5 = {} matched / {} mismatched",
            fixture.file_name().unwrap_or_default().to_string_lossy(),
            stats.total_decoded,
            stats.profile_0,
            stats.profile_1,
            stats.profile_other,
            stats.show_frame_zero,
            stats.visible_md5_matched,
            stats.visible_md5_mismatched,
        );

        assert_eq!(
            shown_idx,
            md5_entries.len(),
            "{}: decoded {shown_idx} shown frames; ledger expects {}",
            fixture.display(),
            md5_entries.len()
        );
        aggregate_show_frame_zero += stats.show_frame_zero;
        total_mismatched += stats.visible_md5_mismatched;
    }

    // G8.0 review (2026-05-21): print aggregate stats BEFORE the asserts
    // so a post-mortem (CI log, local stderr) still gets the numbers
    // even when one of the gates trips.
    println!(
        "real_world.rs aggregate: {} fixture(s), show_frame=0 = {}, \
         visible MD5 mismatches = {}",
        fixtures.len(),
        aggregate_show_frame_zero,
        total_mismatched,
    );

    // Visible MD5 zero-mismatch gate — load-bearing once fixtures
    // are present (HLD V6 §G8 line 1541 "Zero mismatches required").
    assert_eq!(
        total_mismatched, 0,
        "real-world corpus visible MD5 mismatches: {total_mismatched}",
    );

    // Aggregate `show_frame=0` enumeration target (HLD V6 §G8 line
    // 1545: "show_frame=0 count (must be > 0 across the corpus)"). The
    // 30% threshold (line 1519-1520) is a G8.1 corpus-acceptance
    // criterion, not a per-fixture gate — keep this assertion at
    // simply "> 0" so a small curated subset of intra-heavy clips
    // doesn't accidentally fail it.
    assert!(
        aggregate_show_frame_zero > 0,
        "G8 acceptance: aggregate show_frame=0 count is zero across all \
         fixtures — at least one clip must contain hidden (altref) \
         frames. See HLD V6 §G8 line 1545."
    );
}
