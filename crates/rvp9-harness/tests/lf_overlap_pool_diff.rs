//! Scratch-backed tile decode and loop-filter differential test.
//!
//! Each vector decodes once with `threads=1` (serial decode + serial
//! LF — the golden reference) and once with `threads=4` (isolated
//! per-tile scratch frames followed by merge + LF). Per-frame YV12
//! MD5s MUST match.
//!
//! The old output-frame overlap construction has been removed because tile
//! workers use scratch frames: filtering the output before merge would
//! process stale pixels and its whole-plane references could alias.
//!
//! Run via:
//! ```text
//! cargo test -p rvp9-harness --test lf_overlap_pool_diff
//! ```
//!
//! Filtering always follows the tile decode scope and scratch merge.

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

/// Multi-tile vectors exercise scratch decode, merge, then LF.
const MULTI_TILE_VECTORS: &[&str] = &[
    "vp90-2-08-tile_1x2.webm",
    "vp90-2-08-tile_1x4.webm",
    "vp90-2-08-tile_1x8.webm",
];

/// Single-tile vectors keep the serial baseline represented.
const SINGLE_TILE_LF_VECTORS: &[&str] = &["vp90-2-00-quantizer-26.webm", "vp90-2-09-aq2.webm"];

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn scratch_tile_decode_and_lf_match_serial() {
    let dir = vectors_dir();
    // The removed output-frame overlap path must never engage.
    let overlap_frames_pre =
        rvp9_decoder::RVP9_LF_OVERLAP_POOL_FRAMES.load(std::sync::atomic::Ordering::Relaxed);

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
            "{name}: parallel decoded {} shown frames; ledger expects {expected}",
            parallel.len()
        );
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
                 (serial baseline) and threads=4 scratch path \
                 (serial={s:02x?} vs scratch={p:02x?})",
            );
        }
        checked += 1;
    }
    assert_eq!(
        checked,
        SINGLE_TILE_LF_VECTORS.len() + MULTI_TILE_VECTORS.len()
    );

    let overlap_frames_post =
        rvp9_decoder::RVP9_LF_OVERLAP_POOL_FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        overlap_frames_post, overlap_frames_pre,
        "output-frame LF overlap engaged during scratch-backed tile decode"
    );
}
