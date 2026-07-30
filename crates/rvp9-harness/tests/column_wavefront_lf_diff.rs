//! Lever B v2 differential test — exercises the column-wavefront
//! row-MT loopfilter worker (the default for the row-MT LF path
//! after C3).
//!
//! Each vector decodes once with `threads=1` (serial LF path) and
//! once with `threads=4` (the wavefront dispatcher with its current
//! alias-safe one-worker cap). The per-frame YV12 MD5s MUST match.
//!
//! Covers both multi-tile and single-tile fixtures while the
//! dispatcher stays serial until row-bounded plane slices replace
//! whole-plane mutable references.

use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use rvp9_harness::md5_of_yv12_visible;
use rvp9_harness::md5file::read_md5_file;
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

fn decode_vector_md5s(path: &Path, threads: usize) -> Result<Vec<[u8; 16]>, String> {
    let (data, packets) =
        read_packets(path).map_err(|error| format!("container parse: {error}"))?;
    let mut decoder = Decoder::with_threads(NonZeroUsize::new(threads).expect("non-zero"));
    let mut out: Vec<[u8; 16]> = Vec::new();
    for (packet_idx, range) in packets.iter().enumerate() {
        let bytes = range.slice(&data);
        let parsed = decoder
            .parse_packet(bytes)
            .map_err(|error| format!("packet {packet_idx}: parse failed: {error}"))?;
        for (frame_idx, frame) in parsed.into_iter().enumerate() {
            let shown = frame.header.show_frame
                || matches!(
                    frame.header.kind,
                    rvp9_core::bitstream::FrameKind::ShowExistingFrame { .. }
                );
            let buf = decoder.decode_frame(&frame).map_err(|error| {
                format!("packet {packet_idx} frame {frame_idx}: decode failed: {error}")
            })?;
            if shown {
                out.push(md5_of_yv12_visible(&buf));
            }
        }
    }
    Ok(out)
}

fn expected_shown_frames(path: &Path) -> usize {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .expect("vector path has an extension");
    let ledger = path.with_extension(format!("{extension}.md5"));
    read_md5_file(&ledger)
        .unwrap_or_else(|error| panic!("ledger {}: {error}", ledger.display()))
        .len()
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

/// Single-tile vectors exercise the same capped dispatcher.
const SINGLE_TILE_LF_VECTORS: &[&str] = &["vp90-2-00-quantizer-26.webm", "vp90-2-09-aq2.webm"];

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn column_wavefront_lf_matches_serial_on_multi_tile_vectors() {
    // Column-wavefront LF MT is the default after Lever B v2 C3;
    // no env-var setup needed for the t=4 path to take it. We
    // explicitly do NOT set RVP9_DISABLE_LF_MT (would force serial)
    // or RVP9_DISABLE_COLUMN_WAVEFRONT (retired in C4 — legacy
    // worker is gone). The threads=1 baseline still walks the
    // serial LF path (dispatcher's `num_workers >= 2` gate).

    let dir = vectors_dir();
    let mut checked = 0usize;
    for name in SINGLE_TILE_LF_VECTORS
        .iter()
        .chain(MULTI_TILE_VECTORS.iter())
    {
        let path = dir.join(name);
        assert!(path.exists(), "required vector missing: {}", path.display());
        let serial =
            decode_vector_md5s(&path, 1).unwrap_or_else(|error| panic!("{name} serial: {error}"));
        let parallel =
            decode_vector_md5s(&path, 4).unwrap_or_else(|error| panic!("{name} parallel: {error}"));
        let expected = expected_shown_frames(&path);
        assert_eq!(
            serial.len(),
            expected,
            "{name}: serial decoded {} shown frames; ledger expects {expected}",
            serial.len()
        );
        assert_eq!(
            parallel.len(),
            expected,
            "{name}: wavefront decoded {} shown frames; ledger expects {expected}",
            parallel.len()
        );
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
    assert_eq!(
        checked,
        SINGLE_TILE_LF_VECTORS.len() + MULTI_TILE_VECTORS.len()
    );
}
