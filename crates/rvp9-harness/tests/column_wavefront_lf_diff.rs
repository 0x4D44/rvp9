//! Lever B v2 differential test — exercises the column-wavefront
//! row-MT loopfilter worker (the default for the row-MT LF path
//! after C3).
//!
//! Each vector decodes once with `threads=1` (serial LF path) and
//! once with `threads=4` (column-wavefront). The per-frame YV12 MD5s
//! MUST match — proof that the new wavefront worker is
//! bitwise-identical to serial.
//!
//! Covers both multi-tile fixtures (where the wavefront fires on
//! multiple SB rows concurrently) and single-tile fixtures (where
//! the dispatcher's `num_tile_cols == 1` cap reduces `num_lf_workers`
//! to 1, exercising the cap-fallback path).

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

/// Multi-tile vectors that exercise the column-wavefront worker.
/// Vectors here intentionally mirror `row_mt_lf_diff.rs::MULTI_TILE_VECTORS`
/// so any drift between the two workers shows up as a per-vector mismatch
/// in one test or the other.
const MULTI_TILE_VECTORS: &[&str] = &[
    "vp90-2-08-tile_1x2.webm",
    "vp90-2-08-tile_1x4.webm",
    "vp90-2-08-tile_1x8.webm",
];

/// Single-tile vectors that exercise the dispatcher's fallback
/// behaviour. With `num_tile_cols == 1`, the wavefront has no
/// horizontal parallelism — but the worker count is still > 1, so the
/// `cur_sb_col` allocation and the per-row mask-build still run. This
/// catches the "single-tile path crashed because num_lf_workers was
/// 1" failure mode HLD §3.6 documents.
const SINGLE_TILE_LF_VECTORS: &[&str] = &["vp90-2-00-quantizer-26.webm", "vp90-2-09-aq2.webm"];

#[test]
fn column_wavefront_lf_matches_serial_on_multi_tile_vectors() {
    // Column-wavefront LF MT is the default after Lever B v2 C3;
    // no env-var setup needed for the t=4 path to take it. We
    // explicitly do NOT set RVP9_DISABLE_LF_MT (would force serial)
    // or RVP9_DISABLE_COLUMN_WAVEFRONT (retired in C4 — legacy
    // worker is gone). The threads=1 baseline still walks the
    // serial LF path (dispatcher's `num_workers >= 2` gate).

    let dir = vectors_dir();
    if !dir.exists() {
        eprintln!("vectors dir missing — skip");
        return;
    }
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
            "{name}: frame count differs (serial={}, wavefront={})",
            serial.len(),
            parallel.len(),
        );
        for (i, (s, p)) in serial.iter().zip(parallel.iter()).enumerate() {
            assert_eq!(
                s, p,
                "{name} frame {i}: MD5 mismatch between threads=1 and \
                 threads=4 column-wavefront \
                 (serial={s:02x?} vs wavefront={p:02x?})",
            );
        }
        checked += 1;
    }
    assert!(
        checked > 0,
        "column_wavefront_lf_matches_serial_on_multi_tile_vectors: \
         no vectors present — populate tests/conformance/vectors"
    );
}
