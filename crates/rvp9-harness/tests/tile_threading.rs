//! G5 oracles — tile-threaded decode determinism + multi-tile vector
//! conformance at N ∈ {1, 2}.
//!
//! Covers HLD V6 §G5 oracles #1 (conformance at N ∈ {1, 2}), #2 (inter-
//! thread-count determinism), and #4 (per-field FrameCounts merge
//! equality at a basic level — the all-inline first-cut means the math
//! is identical regardless of N, so the oracle is currently a strong
//! invariant rather than a delta check; once the disjoint-plane split
//! lands and `num_workers > 1` actually spawns workers, this oracle
//! becomes load-bearing).

use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use rvp9_harness::md5file::read_md5_file;
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("test bug: n >= 1")
}

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

/// Decode every shown frame of `vector` at thread count `n`, returning
/// the per-frame MD5 hex strings in shown-frame order. `None` slots
/// mean the frame couldn't be decoded (Unimplemented / UnsupportedProfile)
/// — we propagate the gap so the per-N comparator can verify identical
/// failure modes.
fn decode_md5s(vector: &Path, n: NonZeroUsize) -> Vec<Option<String>> {
    let (data, packets) = read_packets(vector).expect("container parse");
    let mut decoder = Decoder::with_threads(n);
    let mut md5s = Vec::new();
    for range in &packets {
        let packet_bytes = range.slice(&data);
        let parsed = match decoder.parse_packet(packet_bytes) {
            Ok(p) => p,
            Err(_) => continue,
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
                        md5s.push(Some(hex16(md5_of_yv12_visible(&buf))));
                    }
                }
                Err(_) => {
                    if shown {
                        md5s.push(None);
                    }
                }
            }
        }
    }
    md5s
}

/// HLD V6 §G5 oracle #1: conformance at N = 2. Re-runs the full
/// conformance corpus (in shape — every vector + every shown frame)
/// through the decoder at `with_threads(2)`, comparing the MD5 of each
/// frame against the .md5 reference. Asserts 0 mismatches.
///
/// The first-cut G5 path runs every tile inline regardless of `n`, so
/// at this snapshot the test is structurally identical to the N=1
/// conformance run. The test still serves a load-bearing role: it
/// pins the `Decoder::with_threads(2)` construction path against any
/// future regression that bifurcates code-paths on `num_threads > 1`
/// (e.g., when actual worker spawning lands, an early bug that
/// produced different per-tile entropy state at N=2 vs N=1 would
/// surface here as a corpus-wide mismatch).
#[test]
fn conformance_at_n_threads_2() {
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
    assert!(!vectors.is_empty(), "no vectors in {}", dir.display());

    let mut total_passed = 0usize;
    let mut total_mismatched = 0usize;
    for vector in &vectors {
        let md5_path = vector.with_extension(format!(
            "{}.md5",
            vector.extension().and_then(|s| s.to_str()).unwrap_or("")
        ));
        let md5_entries = read_md5_file(&md5_path).expect(".md5 parse");
        let got = decode_md5s(vector, nz(2));
        for (idx, got_md5) in got.iter().enumerate() {
            let want = md5_entries.get(idx);
            match (got_md5, want) {
                (Some(g), Some(e)) if e.md5_hex_visible().as_deref() == Some(g.as_str()) => {
                    total_passed += 1;
                }
                (Some(_), _) => {
                    total_mismatched += 1;
                }
                (None, _) => {
                    // Gated/unimplemented at N=2 — same accounting as
                    // the main conformance test (treated as skipped,
                    // not a mismatch).
                }
            }
        }
    }
    assert_eq!(
        total_mismatched, 0,
        "G5 oracle #1: {total_passed} passed, {total_mismatched} mismatched at N=2"
    );
}

/// HLD V6 §G5 oracle #2: inter-thread-count determinism. For each
/// vector in the corpus, `decode(threads=1) == decode(threads=2)` MD5-
/// for-MD5 on every shown frame.
///
/// At the all-inline first cut, this is essentially a tautology — the
/// per-tile loop runs identically at any N. The test stays load-
/// bearing because (a) it exercises the `Decoder::with_threads(N)`
/// construction path for both values, including `WorkerPool::new(N)`
/// and `Drop`, and (b) once worker spawning lands, it becomes the
/// primary parity oracle for the threaded vs single-threaded paths.
#[test]
fn determinism_n1_eq_n2() {
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
    assert!(!vectors.is_empty(), "no vectors in {}", dir.display());

    let mut total_checked = 0usize;
    for vector in &vectors {
        let v1 = decode_md5s(vector, nz(1));
        let v2 = decode_md5s(vector, nz(2));
        assert_eq!(
            v1.len(),
            v2.len(),
            "G5 oracle #2: {} produced different shown-frame counts \
             at N=1 ({} frames) vs N=2 ({} frames). Decode loop \
             bifurcated on thread count.",
            vector.file_name().unwrap_or_default().to_string_lossy(),
            v1.len(),
            v2.len(),
        );
        for (idx, (m1, m2)) in v1.iter().zip(v2.iter()).enumerate() {
            assert_eq!(
                m1,
                m2,
                "G5 oracle #2: {} frame {}: N=1 MD5 {:?} != N=2 MD5 {:?}",
                vector.file_name().unwrap_or_default().to_string_lossy(),
                idx,
                m1,
                m2,
            );
            total_checked += 1;
        }
    }
    assert!(total_checked > 0, "G5 oracle #2: no frames decoded at all");
    eprintln!("G5 oracle #2: {total_checked} shown frames byte-identical at N=1 vs N=2");
}

/// HLD V6 §G5 oracle #2 (focused multi-tile subset). For the three
/// canonical `tile_1xN` vectors, assert the per-frame MD5 is identical
/// at N=1 and N=2. This is the tightest version of the determinism
/// oracle — same as `determinism_n1_eq_n2` but isolated to the multi-
/// tile vectors so a regression localised to the tile dispatch surface
/// is named in the test failure rather than buried inside a single-tile
/// vector report.
#[test]
fn multi_tile_n1_eq_n2() {
    let vectors = [
        "vp90-2-08-tile_1x2.webm",
        "vp90-2-08-tile_1x4.webm",
        "vp90-2-08-tile_1x8.webm",
    ];
    let dir = vectors_dir();
    for name in &vectors {
        let path = dir.join(name);
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            continue;
        }
        let v1 = decode_md5s(&path, nz(1));
        let v2 = decode_md5s(&path, nz(2));
        assert_eq!(
            v1.len(),
            v2.len(),
            "{name}: shown-frame counts differ ({} vs {})",
            v1.len(),
            v2.len()
        );
        for (idx, (m1, m2)) in v1.iter().zip(v2.iter()).enumerate() {
            assert_eq!(
                m1, m2,
                "{name} frame {idx}: N=1 MD5 {m1:?} != N=2 MD5 {m2:?}"
            );
        }
        assert!(!v1.is_empty(), "{name}: no frames decoded");
    }
}

/// Smoke: `Decoder::with_threads(1)` and `Decoder::with_threads(2)`
/// both round-trip through construction + drop without panicking.
/// `WorkerPool::Drop` joins every worker — if any worker is stuck in
/// `recv()` and the channel close path is buggy, this test deadlocks.
#[test]
fn decoder_construct_and_drop_at_n1_and_n2() {
    {
        let _dec = Decoder::with_threads(nz(1));
    }
    {
        let _dec = Decoder::with_threads(nz(2));
    }
    {
        let _dec = Decoder::with_threads(nz(8));
    }
}
