//! Wave 2 C4 differential test — exercises the row-MT loopfilter path.
//!
//! The main conformance harness pins `num_threads = 1` (deterministic),
//! which bypasses the parallel tile-decode dispatch AND the row-MT LF
//! dispatch. This test re-decodes the multi-tile conformance vectors
//! with 1 vs 4 threads and asserts that the per-frame YV12 MD5 matches
//! between the two runs — the row-MT path MUST produce byte-identical
//! output to the serial path.
//!
//! Row-MT is opt-in (Wave 2 outcome — see HLD A/B); this test sets
//! `RVP9_ROW_MT=1` before constructing the multi-thread Decoder so the
//! row-MT dispatching path is actually exercised (without it the
//! parallel-thread decoder would fall through to the legacy serial LF
//! and the differential test would just compare serial-vs-serial).
//!
//! Lever B v2 C2 (HLD §6.2) extends this test to also exercise the
//! new column-wavefront worker path via `RVP9_USE_COLUMN_WAVEFRONT=1`
//! — both paths must produce byte-identical YV12 to serial. Both env
//! vars set together inside the same test process work because
//! `trace_flags()`'s `OnceLock` reads them simultaneously on first
//! access.
//!
//! IMPORTANT: the env vars are read once via `OnceLock` on the first
//! `trace_flags()` access. The harness `set_var` BEFORE constructing
//! the Decoder, so the OnceLock initialises with both env vars set.
//! If another test in the same process inits `trace_flags` first
//! WITHOUT the vars set, this test will silently skip the new paths.
//! The `tests/` directory layout (one process per `--test`) means
//! this file runs in isolation.

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

/// Decode a vector and return one MD5 (16 bytes) per shown frame.
fn decode_vector_md5s(path: &Path, threads: usize) -> Vec<[u8; 16]> {
    let (data, packets) = read_packets(path).expect("container parse");
    let mut decoder = Decoder::with_threads(NonZeroUsize::new(threads).expect("non-zero"));
    let mut out: Vec<[u8; 16]> = Vec::new();
    for range in &packets {
        let bytes = range.slice(&data);
        let Ok(parsed) = decoder.parse_packet(bytes) else {
            // Truncated / header parse failure — caller decides; we
            // append a sentinel so the run lengths line up.
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

/// Multi-tile vectors that exercise the parallel tile-decode + row-MT
/// LF dispatch at `num_threads > 1`. These are the load-bearing
/// fixtures for the C4 differential check.
const MULTI_TILE_VECTORS: &[&str] = &[
    "vp90-2-08-tile_1x2.webm",
    "vp90-2-08-tile_1x4.webm",
    "vp90-2-08-tile_1x8.webm",
];

#[test]
fn row_mt_lf_matches_serial_on_multi_tile_vectors() {
    // Lever B v2 C3 flipped the dispatcher's LF-worker default to
    // column-wavefront. To exercise the LEGACY row-barrier worker
    // (the original Wave 2 path) this test sets
    // `RVP9_DISABLE_COLUMN_WAVEFRONT=1`. With MT now default-on
    // (no env var needed) the dispatcher routes the t=4 path to
    // `run_row_mt_lf_worker`, matching this test's historical
    // scope. The new worker has its own diff test
    // (`column_wavefront_lf_diff.rs`).
    //
    // `RVP9_ROW_MT=1` is set too for compatibility with the env-var
    // OnceLock that other tests may have inited (idempotent with
    // the new MT default).
    // SAFETY: tests in `tests/` are single-threaded by default in
    // cargo's harness; no other thread reads env vars at this point.
    unsafe { std::env::set_var("RVP9_ROW_MT", "1") };
    unsafe { std::env::set_var("RVP9_DISABLE_COLUMN_WAVEFRONT", "1") };

    let dir = vectors_dir();
    if !dir.exists() {
        eprintln!("vectors dir missing — skip");
        return;
    }
    for name in MULTI_TILE_VECTORS {
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
                "{name} frame {i}: MD5 mismatch between threads=1 and threads=4 \
                 (serial={s:02x?} vs row-MT={p:02x?})",
            );
        }
    }
}
