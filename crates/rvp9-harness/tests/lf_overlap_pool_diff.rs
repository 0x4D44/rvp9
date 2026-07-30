//! Scratch-backed tile decode and loop-filter differential test.
//!
//! Each vector decodes once with `threads=1` (serial decode + serial
//! LF — the golden reference) and once with `threads=4` (isolated
//! per-tile scratch frames followed by merge + LF). Per-frame YV12
//! MD5s MUST match.
//!
//! The old output-frame overlap path remains disabled while tile
//! workers use scratch frames: filtering the output before merge would
//! process stale pixels and its whole-plane references could alias.
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

/// Multi-tile vectors exercise scratch decode, merge, then LF.
const MULTI_TILE_VECTORS: &[&str] = &[
    "vp90-2-08-tile_1x2.webm",
    "vp90-2-08-tile_1x4.webm",
    "vp90-2-08-tile_1x8.webm",
];

/// Single-tile vectors keep the serial baseline represented.
const SINGLE_TILE_LF_VECTORS: &[&str] = &["vp90-2-00-quantizer-26.webm", "vp90-2-09-aq2.webm"];

#[test]
fn scratch_tile_decode_and_lf_match_serial() {
    let dir = vectors_dir();
    if !dir.exists() {
        eprintln!("vectors dir missing — skip");
        return;
    }
    // The unsafe output-frame overlap path must remain dormant.
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
                 (serial baseline) and threads=4 scratch path \
                 (serial={s:02x?} vs scratch={p:02x?})",
            );
        }
        checked += 1;
    }
    if checked == 0 {
        eprintln!(
            "no scratch/LF vectors present — \
             run scripts/fetch-vectors.sh to exercise this optional oracle"
        );
        return;
    }

    let overlap_frames_post =
        rvp9_decoder::RVP9_LF_OVERLAP_POOL_FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        overlap_frames_post, overlap_frames_pre,
        "output-frame LF overlap engaged during scratch-backed tile decode"
    );
}
