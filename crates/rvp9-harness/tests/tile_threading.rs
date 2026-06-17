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

/// G5.1 oracle: the parallel dispatch path MUST actually fire on a
/// multi-tile vector at N=2 (non-tautological N=2 conformance).
///
/// Reads the process-wide `RVP9_G5_1_PARALLEL_JOBS` counter before and
/// after decoding a `tile_1x2` vector at N=2; asserts the delta is
/// `num_shown_frames * 2` (= 2 tile jobs per frame). If the eligibility
/// gate misfires or the dispatch path silently falls back to inline,
/// the counter doesn't bump and the assert fires.
#[test]
fn multi_tile_dispatch_uses_worker_pool() {
    use std::sync::atomic::Ordering;
    let dir = vectors_dir();
    let path = dir.join("vp90-2-08-tile_1x2.webm");
    if !path.exists() {
        eprintln!("skip: {} not present", path.display());
        return;
    }
    let before = rvp9_decoder::RVP9_G5_1_PARALLEL_JOBS.load(Ordering::Relaxed);
    let md5s = decode_md5s(&path, nz(2));
    let after = rvp9_decoder::RVP9_G5_1_PARALLEL_JOBS.load(Ordering::Relaxed);
    let delta = after.wrapping_sub(before);
    let n_frames = md5s.iter().filter(|m| m.is_some()).count() as u32;
    assert!(n_frames > 0, "tile_1x2 produced 0 frames — corpus broken");
    // 2 tiles per frame in `tile_1x2`. Expect delta >= 2 * n_frames
    // (=) ; "exactly" if every shown frame is a multi-tile frame
    // (which holds for tile_1x2 — every frame has the same tile config).
    assert!(
        delta >= 2 * n_frames,
        "G5.1 parallel dispatch under-fired: delta={delta}, expected >= {} (2 * {} frames)",
        2 * n_frames,
        n_frames,
    );
    eprintln!(
        "G5.1: {} parallel tile jobs fired across {} frames (= avg {:.1} jobs/frame)",
        delta,
        n_frames,
        delta as f64 / n_frames as f64
    );
}

/// G5.2 oracle: the per-tile scratch pool persists across frames
/// and stops allocating once the steady-state tile count is reached.
///
/// Pattern: decode the SAME multi-tile vector (tile_1x8 — 8 tiles)
/// frame-by-frame through a single `Decoder` instance. After the
/// first multi-tile frame, the pool holds `n_tiles` buffers. Every
/// subsequent multi-tile frame's `take(n_tiles, …)` must be satisfied
/// from the cache, so the post-decode `tile_scratch_pool_buffer_count`
/// MUST remain at the steady-state value (no shrinkage; no growth).
///
/// We also snapshot the process-wide [`RVP9_G5_2_SCRATCH_ALLOCS`]
/// counter delta per frame as a secondary signal — but we accept
/// non-zero increments from concurrent tests (this binary's tests
/// run in parallel and other tests use multi-tile decoders that
/// also bump this counter). The pool-buffer-count assertion is
/// the load-bearing one; the counter delta is reported for
/// diagnostic purposes.
///
/// Vector choice: `tile_1x8.webm` has the largest tile count in the
/// conformance corpus that we have hands-on multi-tile parallel-
/// dispatch eligibility for, so this catches both growth (from 0 to
/// 8) and steady-state-no-growth (from frame 2 onwards).
#[test]
fn tile_scratch_pool_warm_no_alloc() {
    use std::sync::atomic::Ordering;
    let dir = vectors_dir();
    let path = dir.join("vp90-2-08-tile_1x8.webm");
    if !path.exists() {
        eprintln!("skip: {} not present", path.display());
        return;
    }
    let (data, packets) = rvp9_harness::container::read_packets(&path).expect("container parse");
    let mut decoder = Decoder::with_threads(nz(2));

    // Before any decode, the pool should be empty.
    assert_eq!(
        decoder.tile_scratch_pool_buffer_count(),
        0,
        "fresh decoder pool should start empty"
    );

    let mut frame_idx = 0usize;
    let mut alloc_delta_by_frame: Vec<u32> = Vec::new();
    let mut pool_count_by_frame: Vec<usize> = Vec::new();

    for range in &packets {
        let packet_bytes = range.slice(&data);
        let parsed = match decoder.parse_packet(packet_bytes) {
            Ok(p) => p,
            Err(_) => continue,
        };
        for frame in parsed {
            let before = rvp9_core::frame::RVP9_G5_2_SCRATCH_ALLOCS.load(Ordering::Relaxed);
            let _ = decoder.decode_frame(&frame);
            let after = rvp9_core::frame::RVP9_G5_2_SCRATCH_ALLOCS.load(Ordering::Relaxed);
            alloc_delta_by_frame.push(after.wrapping_sub(before));
            pool_count_by_frame.push(decoder.tile_scratch_pool_buffer_count());
            frame_idx += 1;
            // Bound the test runtime — we only need a few frames
            // past warmup to make the assertion.
            if frame_idx >= 5 {
                break;
            }
        }
        if frame_idx >= 5 {
            break;
        }
    }

    assert!(
        frame_idx >= 2,
        "needed at least 2 frames from tile_1x8.webm to test pool warmup"
    );

    // After frame 1: pool holds n_tiles_steady buffers (the take
    // grew it; the post-merge put restored it). tile_1x8 has 8
    // tiles ⇒ 8 buffers.
    let n_tiles_steady = pool_count_by_frame[0];
    assert!(
        n_tiles_steady >= 2,
        "tile_1x8 should grow pool to ≥ 2 buffers after frame 1, was {n_tiles_steady}"
    );

    // Frames 2+: pool is warm. The pool-buffer-count post-decode
    // MUST remain at n_tiles_steady (no growth — take(n) is fully
    // satisfied from the cache; no shrinkage — put restores all
    // borrowed buffers).
    for i in 1..pool_count_by_frame.len() {
        assert_eq!(
            pool_count_by_frame[i], n_tiles_steady,
            "G5.2: frame {} pool count {} != steady-state {}. \
             Per-frame pool counts: {:?}",
            i, pool_count_by_frame[i], n_tiles_steady, pool_count_by_frame,
        );
    }

    // Pool size never shrinks: monotonic. (Already covered by the
    // equality check above; kept as a defensive invariant.)
    for (i, &count) in pool_count_by_frame.iter().enumerate() {
        assert!(
            count >= n_tiles_steady,
            "G5.2: frame {i} pool count {count} < steady-state {n_tiles_steady} (pool shrank!)",
        );
    }

    eprintln!(
        "G5.2: pool warmed to {} buffers, {} steady-state frames at constant pool size. \
         (Counter deltas: {:?} — non-zero values past frame 0 are from concurrent tests, \
         not from this decoder's pool.)",
        n_tiles_steady,
        pool_count_by_frame.len() - 1,
        &alloc_delta_by_frame
    );
}

/// G5.1 oracle: re-running the same multi-tile vector twice in
/// succession produces bit-identical MD5s on every frame. Catches
/// thread-scheduling non-determinism — if the merge order varies
/// between runs (e.g. results collected in completion order instead
/// of submission order), the running counts diverge after the first
/// frame and the second run's MD5s drift.
#[test]
fn multi_tile_run_to_run_determinism() {
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
        let run1 = decode_md5s(&path, nz(2));
        let run2 = decode_md5s(&path, nz(2));
        assert_eq!(
            run1, run2,
            "{name}: G5.1 run-to-run determinism broken — same vector \
             produced different MD5s on consecutive decodes",
        );
        assert!(!run1.is_empty(), "{name}: no frames decoded");
    }
}
