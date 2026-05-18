//! Stage 3 Task E.2 — single-threaded multi-tile decode regression
//! tests. Before E.2, the decoder rejected any frame with
//! `tile_cols > 1 || tile_rows > 1` via
//! `TileDecodeError::MultiTileUnsupported`. E.2 wires up the
//! iteration over per-tile byte buffers (parsed by libvpx-style
//! 4-byte BE size prefixes) and calls `decode_tile_*` once per tile
//! with its own fresh `BoolReader` / `TileDecodeCtx` / `TileCoefCtx`.
//! These tests assert the gate is gone for the canonical conformance
//! multi-tile vectors (1x2 / 1x4 / 1x8).

use rvp9_decoder::{DecodeError, Decoder, tile::TileDecodeError};
use rvp9_harness::container::read_packets;
use std::path::PathBuf;

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

fn assert_vector_decodes_without_multi_tile_reject(vector_name: &str) {
    let path = vectors_dir().join(vector_name);
    if !path.exists() {
        eprintln!(
            "skip: {} not present (run scripts/fetch-vectors.ps1)",
            path.display()
        );
        return;
    }
    let (data, packets) = read_packets(&path).expect("container parse");
    let mut dec = Decoder::new();
    let mut decoded = 0usize;
    for range in &packets {
        let parsed = dec.parse_packet(range.slice(&data)).expect("packet parse");
        for frame in parsed {
            match dec.decode_frame(&frame) {
                Ok(_) => decoded += 1,
                // MultiTileUnsupported is IMPOSSIBLE after E.2 — any
                // occurrence is a regression.
                Err(DecodeError::TileDecode(TileDecodeError::MultiTileUnsupported {
                    ..
                })) => {
                    panic!(
                        "{vector_name}: regression — multi-tile decode \
                         still hits MultiTileUnsupported after E.2"
                    );
                }
                // Other errors (Unimplemented, FrameAlloc, etc.) are
                // acceptable here — this test only asserts the
                // multi-tile gate is gone, not full bit-exactness
                // (the conformance gate tracks that separately).
                Err(_) => {}
            }
        }
    }
    assert!(
        decoded > 0,
        "{vector_name}: expected at least one frame to decode end-to-end"
    );
}

/// E.2.b — 1x2 (one row, two cols) tile vector. 1920×1080 inter-
/// prediction stream; previously skipped at the multi-tile gate.
#[test]
fn vp90_2_08_tile_1x2_decodes_after_e2() {
    assert_vector_decodes_without_multi_tile_reject("vp90-2-08-tile_1x2.webm");
}

/// E.2.b — 1x4 (one row, four cols) tile vector.
#[test]
fn vp90_2_08_tile_1x4_decodes_after_e2() {
    assert_vector_decodes_without_multi_tile_reject("vp90-2-08-tile_1x4.webm");
}

/// E.2.b — 1x8 (one row, eight cols) tile vector.
#[test]
fn vp90_2_08_tile_1x8_decodes_after_e2() {
    assert_vector_decodes_without_multi_tile_reject("vp90-2-08-tile_1x8.webm");
}
