//! Phase 1 separate-LF-pool overlap differential test (HLD
//! `2026.05.27 - HLD - Separate LF thread pool.md` §7.1).
//!
//! Each vector decodes once with `threads=1` (serial decode + serial
//! LF — the golden reference) and once with `threads=4` (the overlap
//! path, default ON after the +8.14% mean PGO win on testsrc t=4). The
//! per-frame YV12 MD5s MUST match — proof that the new LF pool's L1
//! gate produces bitwise-identical output to the legacy two-scope path.
//!
//! Failure mode this catches: L1 off-by-one — LF starting one SB-row
//! too early and filtering pixels that a future intra-prediction read
//! still needs unfiltered (HLD §4.3 Invariant L1).
//!
//! Run via:
//! ```text
//! cargo test -p rvp9-harness --test lf_overlap_pool_diff
//! ```
//!
//! To force the legacy two-scope path (for differential against
//! that explicitly), set `RVP9_DISABLE_LF_OVERLAP_POOL=1`.

use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use rvp9_harness::md5_of_yv12_visible;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

fn vectors_dir() -> PathBuf {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("tests")
        .join("conformance")
        .join("vectors")
}

fn decode_vector_md5s(path: &Path, threads: usize) -> Vec<[u8; 16]> {
    let (data, packets) = read_packets(path).expect("container parse");
    let mut decoder = Decoder::with_threads(NonZeroUsize::new(threads).expect("non-zero"));
    let mut out: Vec<[u8; 16]> = Vec::new();
    for range in &packets {
        let bytes = range.slice(&data);
        let Ok(parsed) = decoder.parse_packet(bytes) else {
            out.push([0u8; 16]);
            continue;
        };
        for frame in parsed {
            let shown = frame.header.show_frame
                || matches!(
                    frame.header.kind,
                    rvp9_core::bitstream::FrameKind::ShowExistingFrame { .. }
                );
            match decoder.decode_frame(&frame) {
                Ok(buf) => {
                    if shown {
                        out.push(md5_of_yv12_visible(&buf));
                    }
                }
                Err(_) => {
                    if shown {
                        out.push([0u8; 16]);
                    }
                }
            }
        }
    }
    out
}

/// Multi-tile vectors — exercise the overlap path's primary target
/// (column-wavefront LF gated on `decode_progress[r+1] >=
/// num_tile_cols`). Mirrors the legacy `column_wavefront_lf_diff.rs`
/// vector list so the two tests catch any drift between paths.
const MULTI_TILE_VECTORS: &[&str] = &[
    "vp90-2-08-tile_1x2.webm",
    "vp90-2-08-tile_1x4.webm",
    "vp90-2-08-tile_1x8.webm",
];

/// Single-tile vectors — eligibility check ineligibility (since
/// `tile_rows == 1` AND `tile_cols >= 1` both hold; single-tile is
/// covered with `num_tile_cols = 1`). The L1 gate trivially passes
/// since each row only has one tile bumping it. Confirms the overlap
/// path doesn't divergently corrupt single-tile output.
const SINGLE_TILE_LF_VECTORS: &[&str] = &["vp90-2-00-quantizer-26.webm", "vp90-2-09-aq2.webm"];

#[test]
fn lf_overlap_pool_matches_serial() {
    let dir = vectors_dir();
    if !dir.exists() {
        eprintln!("vectors dir missing — skip");
        return;
    }
    // Snapshot the overlap-engaged counter so we can later assert the
    // overlap path actually ran (i.e. the test wasn't silently
    // pass-through on the legacy path because the env var was unset).
    let overlap_frames_pre =
        rvp9_decoder::RVP9_LF_OVERLAP_POOL_FRAMES.load(std::sync::atomic::Ordering::Relaxed);

    let mut checked = 0usize;
    for name in SINGLE_TILE_LF_VECTORS
        .iter()
        .chain(MULTI_TILE_VECTORS.iter())
    {
        let path = dir.join(name);
        if !path.exists() {
            eprintln!("missing {name} — skip");
            continue;
        }
        let serial = decode_vector_md5s(&path, 1);
        let parallel = decode_vector_md5s(&path, 4);
        assert_eq!(
            serial.len(),
            parallel.len(),
            "{name}: frame count differs (serial={}, parallel={})",
            serial.len(),
            parallel.len(),
        );
        for (i, (s, p)) in serial.iter().zip(parallel.iter()).enumerate() {
            assert_eq!(
                s, p,
                "{name} frame {i}: MD5 mismatch between threads=1 \
                 (serial baseline) and threads=4 overlap path \
                 (serial={s:02x?} vs overlap={p:02x?})",
            );
        }
        checked += 1;
    }
    if checked == 0 {
        eprintln!(
            "no LF-overlap vectors present — \
             run scripts/fetch-vectors.sh to exercise this optional oracle"
        );
        return;
    }

    // The overlap path is default ON; the threads=4 decode of any
    // multi-tile fixture above MUST have engaged it — verifies the
    // test is actually exercising the new code, not silently passing
    // on the legacy two-scope path. Skip the check only when the
    // operator explicitly disabled it.
    let overlap_frames_post =
        rvp9_decoder::RVP9_LF_OVERLAP_POOL_FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    if std::env::var("RVP9_DISABLE_LF_OVERLAP_POOL").is_err() {
        assert!(
            overlap_frames_post > overlap_frames_pre,
            "overlap path is default ON but never engaged \
             (counter: pre={overlap_frames_pre} post={overlap_frames_post}). \
             Either the LF pool construction skipped (cores-available \
             math), or the eligibility check rejected every vector.",
        );
    }
}
