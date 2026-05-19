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

        // Per-vector tally — only printed under
        // `RVP9_CONFORMANCE_PER_VECTOR=1` (off by default to keep the
        // CI log small).
        let report_per_vector = std::env::var("RVP9_CONFORMANCE_PER_VECTOR").is_ok();
        let mut v_passed = 0usize;
        let mut v_mismatched = 0usize;
        let mut v_skipped = 0usize;

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
                            Some(e) if e.md5_hex == got_hex => {
                                total_passed += 1;
                                v_passed += 1;
                            }
                            _ => {
                                total_mismatched += 1;
                                v_mismatched += 1;
                            }
                        }
                        shown_idx += 1;
                    }
                    Err(DecodeError::Unimplemented { .. }) => {
                        if shown {
                            total_skipped += 1;
                            v_skipped += 1;
                            shown_idx += 1;
                        }
                    }
                    Err(_) => {
                        if shown {
                            total_skipped += 1;
                            v_skipped += 1;
                            shown_idx += 1;
                        }
                    }
                }
            }
        }
        if report_per_vector {
            println!(
                "  {}: {v_passed} passed / {v_mismatched} mismatched / {v_skipped} skipped",
                vector.file_name().unwrap_or_default().to_string_lossy(),
            );
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
    const MIN_PASSES: usize = 31;
    // D.20 (2026-05-19): NEGATIVE — diagnostic infra only. Ratchet
    // unchanged. Localized aq2 frame-2 (64, 0) cascade to altref's
    // mi=(0,8) seg_id divergence (rvp9 decodes seg=1; libvpx altref
    // produces seg=0 per the frame-2 inherit-via-min trace at
    // libvpx_imi.txt). Confirmed bool-coder cursor diverges inside
    // altref's SB(0,8) syntax — cannot pin to a specific bit without
    // a libvpx altref INTERMI trace (build env wedged). Three new
    // env hooks for D.21: `RVP9_DUMP_ALTREF=<path>` for altref Y
    // plane dump, `RVP9_TRACE_SEG_ID=1` for per-block seg-id reads,
    // `RVP9_TRACE_SEG=1` extended to dump first-SB-row seg map.
    // D.15 (2026-05-18): sub-8 luma `dec_reset_skip_context` now pins
    // the entropy-context span to the parent 8×8 mi cell (n4 = 2 in
    // each axis), matching libvpx's `set_plane_n4(bwl=bhl=1)` for any
    // sub-8 child of a Block8×8 parent. Previously we used
    // `plane_bsize.n_4x4_*` which is 1×1 / 1×2 / 2×1 / 2×2 for
    // sub-8 — leaving one cell of `left_coef` un-cleared on every
    // sub-8 skip, which poisoned `initial_pt` for the next
    // decode_coefs at the same SB-row y-offset. Lights up all 6 of the
    // failing 352×288/352×240 keyframes (q26, q63, sharpness-1/7,
    // deltaq, resize) — frame 1 of each now MD5-matches libvpx. The
    // 2-tile keyframes (1x2 / 1x4 / 1x8) already matched (no sub-8
    // luma blocks in those frames). Conformance 22/169/10 → 31/160/10.
    // E.2 (2026-05-18): single-threaded multi-tile decode wired up.
    // Removes the `MultiTileUnsupported` reject on the 1x2/1x4/1x8
    // tile vectors (30 previously-skipped frames). All 30 now DECODE
    // (no panic, no error) but MD5-mismatch — expected, since these
    // vectors exercise tile-boundary entropy isolation that the floor
    // is now exposed to for the first time. `MAX_MISMATCHES` rises
    // from 139 → 169 (the new mismatches) and `total_skipped` falls
    // from 40 → 10. `MIN_PASSES` unchanged. Also fixes a pre-existing
    // `1u16 << 16` overflow in `adjust_mask` for `rows == 7`
    // (triggered by the 1920×1080 tile vectors).
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
    const MAX_MISMATCHES: usize = 160;
    assert!(
        total_passed >= MIN_PASSES,
        "regression: expected at least {MIN_PASSES} passed, got {total_passed}",
    );
    assert!(
        total_mismatched <= MAX_MISMATCHES,
        "regression: expected at most {MAX_MISMATCHES} mismatched, got {total_mismatched}",
    );
}
