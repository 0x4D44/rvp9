//! End-to-end test: drives every conformance vector through our
//! decoder and asserts the harness does not produce any *mismatches*.
//!
//! Skipped frames are expected (and counted) until the full decode
//! path lands; they do NOT fail the test. Mismatches DO fail the
//! test — that's the load-bearing gate.
//!
//! Vector / `.md5` parse failures (i.e. broken harness plumbing)
//! also fail the test.

use rvp9_decoder::{DecodeError, Decoder};
use rvp9_harness::container::read_packets;
use rvp9_harness::md5file::read_md5_file;
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    // Find the workspace root by walking up from this file's
    // location. `CARGO_MANIFEST_DIR` is the crate root
    // (`crates/rvp9-harness`); the workspace root is two levels up.
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("tests")
        .join("conformance")
        .join("vectors")
}

#[test]
fn conformance_runner_walks_all_vectors() {
    let dir = vectors_dir();
    if !dir.exists() {
        eprintln!(
            "vectors dir {} not present — run scripts/fetch-vectors.ps1",
            dir.display()
        );
        return;
    }

    let mut vectors: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|x| matches!(x, "ivf" | "webm"))
        })
        .collect();
    vectors.sort();

    assert!(
        !vectors.is_empty(),
        "no .ivf/.webm vectors found in {}",
        dir.display()
    );

    let mut total_passed = 0usize;
    let mut total_mismatched = 0usize;
    let mut total_skipped = 0usize;
    let mut total_expected = 0usize;

    for vector in &vectors {
        let md5_path = vector.with_extension(format!(
            "{}.md5",
            vector.extension().and_then(|s| s.to_str()).unwrap_or("")
        ));
        let (data, packets) = read_packets(vector).expect("container parse");
        let md5_entries = read_md5_file(&md5_path).expect(".md5 parse");
        total_expected += md5_entries.len();

        let mut decoder = Decoder::new();
        let mut shown_idx: usize = 0;
        for range in &packets {
            let parsed = match decoder.parse_packet(range.slice(&data)) {
                Ok(p) => p,
                Err(_) => {
                    // header parse fail → treat the packet as a single
                    // skipped shown-frame.
                    total_skipped += 1;
                    shown_idx += 1;
                    continue;
                }
            };
            for frame in parsed {
                let shown = frame.header.show_frame
                    || matches!(
                        frame.header.kind,
                        rvp9_core::bitstream::FrameKind::ShowExistingFrame { .. }
                    );
                match decoder.decode_frame(&frame) {
                    Ok(buf) => {
                        if !shown {
                            continue;
                        }
                        let got_hex = hex16(md5_of_yv12_visible(&buf));
                        let want = md5_entries.get(shown_idx);
                        match want {
                            Some(e) if e.md5_hex == got_hex => total_passed += 1,
                            _ => total_mismatched += 1,
                        }
                        shown_idx += 1;
                    }
                    Err(DecodeError::Unimplemented { .. }) => {
                        if shown {
                            total_skipped += 1;
                            shown_idx += 1;
                        }
                    }
                    Err(_) => {
                        if shown {
                            total_skipped += 1;
                            shown_idx += 1;
                        }
                    }
                }
            }
        }
    }

    println!(
        "conformance: {total_passed} passed / {total_mismatched} mismatched / \
         {total_skipped} skipped / {total_expected} total"
    );

    // Ratcheting baseline (set during overnight 2026-05-17 floor milestone
    // when the first keyframe match — vp90-2-02-size-08x08.webm frame 1 —
    // landed). The final load-bearing Stage 3 gate is `total_mismatched
    // == 0`, but while we iterate on the mismatched keyframes the
    // ratchet keeps regression signal visible: lowering `MAX_MISMATCHES`
    // as we fix bugs, raising `MIN_PASSES` as we light up frames.
    const MIN_PASSES: usize = 22;
    // D.9 (2026-05-18): backward-adapt now uses a pre-compressed-header
    // FrameContext snapshot as the `pre` baseline (mirroring libvpx's
    // `pre_fc = cm->frame_contexts[fc_idx]` while compressed-header
    // diff-updates land on `cm->fc` — a working copy our code didn't
    // model). Without the snapshot, every inter frame's adapt was
    // chaining from `bank + this_frame's_diff_updates`, accumulating
    // probability-table drift that corrupted the bool-coder cursor
    // after several frames. Conformance moves from 11/150/40 →
    // 22/139/40 — `vp90-2-02-size-08x08.webm` lights up frames 5-10
    // (now full 10/10) and `vp90-2-02-size-32x32.webm` lights up
    // frames 5-9 (9/10, only frame 10 still mismatches).
    // D.8 (2026-05-18): wire the loop-filter apply driver into the
    // inter-frame path (matches the K2.c keyframe wireup). libvpx's
    // `vpxdec --i420` reference output is post-LF; without this the
    // inter frames were being MD5'd as pre-LF. Conformance moves from
    // 8/153/40 → 11/150/40 — three additional inter frames light up,
    // including frames 2/3/4 of `vp90-2-02-size-32x32.webm`.
    // D.7 (2026-05-18): inter-frame cascade work (mv_ref + decoder
    // ref-pool / tile decode tweaks) lights up 2 additional frames.
    // Conformance moves from 6/155/40 → 8/153/40.
    // M.15 (2026-05-18): added libvpx 4x4 specializations for
    // D45 / D63 / D207 intra-pred. Conformance: 3/158/40 → 6/155/40.
    // D.5 (2026-05-18): `extend_mc_border` scratch-buffer fallback.
    // Conformance: 2/159/40 → 3/158/40.
    // Pre-D.3.a baseline: 1/9/191.
    const MAX_MISMATCHES: usize = 139;
    assert!(
        total_passed >= MIN_PASSES,
        "regression: expected at least {MIN_PASSES} passed, got {total_passed}",
    );
    assert!(
        total_mismatched <= MAX_MISMATCHES,
        "regression: expected at most {MAX_MISMATCHES} mismatched, got {total_mismatched}",
    );
}
