//! Stage 11 (coverage push) — per-vector conformance tests.
//!
//! The big `rvp9-harness::tests::conformance_runner_walks_all_vectors`
//! integration test is excluded from `cargo llvm-cov --exclude
//! rvp9-harness` runs, so the production code paths it exercises
//! (per-profile, multi-tile, show-existing-frame, resize, sub-8 MV)
//! don't currently contribute branch coverage. Mirror it here, in a
//! crate that IS in the coverage scope, with one test per vector so a
//! single regression points at a single named test.
//!
//! Each test decodes one vector and asserts the per-frame visible MD5
//! matches the sibling `.md5` ledger libvpx shipped alongside it.
//!
//! This file deliberately depends only on `rvp9-io` (not
//! `rvp9-harness`) to avoid a dev-dep cycle through `rvp9-harness →
//! rvp9-decoder` (regular dep).

use rvp9_core::bitstream::FrameKind;
use rvp9_decoder::{DecodeError, Decoder};
use rvp9_io::container::read_packets;
use rvp9_io::md5::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
use std::path::PathBuf;

/// Locate the workspace's `tests/conformance/vectors/` directory.
///
/// The decoder crate sits at `crates/rvp9-decoder/`; walk two levels
/// up to reach the workspace root, then descend into `tests/...`.
fn vectors_dir() -> PathBuf {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .expect("crate root has parent.parent")
        .join("tests")
        .join("conformance")
        .join("vectors")
}

/// Inline trivial `.md5` ledger parser — one entry per displayed
/// frame, holding only the 32-char hex digest (sufficient for the
/// equality oracle below).
///
/// Format (libvpx's `vpxdec --md5`):
/// ```text
/// <32-hex>  <name>-<WxH>-<NNNN>.i420
/// ```
fn read_md5_hex_lines(path: &PathBuf) -> Vec<String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!(".md5 read {}: {e}", path.display()));
    let mut hexes = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (h, _) = line
            .split_once("  ")
            .unwrap_or_else(|| panic!(".md5 line missing 2-space separator: {line:?}"));
        assert_eq!(h.len(), 32, "expected 32-char md5 hex on line {line:?}");
        hexes.push(h.to_ascii_lowercase());
    }
    hexes
}

/// Common driver: decode `name.webm` (or `.ivf`) and assert every
/// displayed frame's visible MD5 matches the sibling ledger.
///
/// When the vectors directory has no vector files, the build script
/// marks every test in this file ignored instead of letting it report
/// a false success. If any vector is present, all tests run so a
/// partial corpus fails on the first missing vector or ledger.
fn drive_vector(name: &str) {
    let dir = vectors_dir();
    let vec_path = dir.join(name);
    assert!(
        vec_path.exists(),
        "vector missing from a partially populated corpus: {}",
        vec_path.display()
    );
    let ext = vec_path
        .extension()
        .and_then(|s| s.to_str())
        .expect("vector path has extension");
    let md5_path = vec_path.with_extension(format!("{ext}.md5"));
    assert!(
        md5_path.exists(),
        "ledger missing alongside vector: {}",
        md5_path.display()
    );

    let want = read_md5_hex_lines(&md5_path);
    let (data, packets) = read_packets(&vec_path).expect("container parse");

    let mut decoder = Decoder::with_threads(NonZeroUsize::new(1).expect("1 is non-zero"));
    let mut shown_idx: usize = 0;
    for range in &packets {
        let parsed = match decoder.parse_packet(range.slice(&data)) {
            Ok(p) => p,
            Err(e) => panic!("parse_packet failed on {name}: {e:?}"),
        };
        for frame in parsed {
            let shown = frame.header.show_frame
                || matches!(frame.header.kind, FrameKind::ShowExistingFrame { .. });
            match decoder.decode_frame(&frame) {
                Ok(buf) => {
                    if !shown {
                        continue;
                    }
                    let got = hex16(md5_of_yv12_visible(&buf));
                    let expected = want.get(shown_idx).unwrap_or_else(|| {
                        panic!(
                            "{name}: ran out of ledger entries at shown_idx={shown_idx}; \
                             got {got}"
                        )
                    });
                    assert_eq!(
                        &got, expected,
                        "{name}: visible-MD5 mismatch at shown_idx={shown_idx}"
                    );
                    shown_idx += 1;
                }
                Err(DecodeError::Unimplemented { .. }) => {
                    panic!(
                        "{name}: Unimplemented decode at shown_idx={shown_idx} — \
                         Stage 11 expects every vector in this curated set to \
                         decode end-to-end"
                    );
                }
                Err(e) => panic!("{name}: decode_frame error at shown_idx={shown_idx}: {e:?}"),
            }
        }
    }
    assert_eq!(
        shown_idx,
        want.len(),
        "{name}: decoded {shown_idx} shown frames; ledger expects {}",
        want.len()
    );
}

// ---------------------------------------------------------------------
// Per-vector tests — one per #[test] so a regression points at the
// specific vector. The chosen subset covers:
//
//   * Profile 0 — quantizer sweep (small/mid/large)        (×3)
//   * Loop-filter sharpness sweep                           (×2)
//   * Edge-size frames (8×8 and 32×32)                      (×2)
//   * Segmentation + delta-Q                                (×1)
//   * Mid-stream resize (the only IVF vector)               (×1)
//   * Multi-tile (1×2 / 1×4 / 1×8 cols)                     (×3)
//   * AQ mode 2 (segmentation + altref)                     (×1)
//   * Show-existing-frame                                   (×1)
//
// Total: 14 vectors — every shipped conformance vector. Each is a
// distinct exercise of the decoder state machine.
// ---------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_quantizer_00() {
    drive_vector("vp90-2-00-quantizer-00.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_quantizer_26() {
    drive_vector("vp90-2-00-quantizer-26.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_quantizer_63() {
    drive_vector("vp90-2-00-quantizer-63.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_sharpness_1() {
    drive_vector("vp90-2-01-sharpness-1.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_sharpness_7() {
    drive_vector("vp90-2-01-sharpness-7.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_size_08x08() {
    drive_vector("vp90-2-02-size-08x08.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_size_32x32() {
    drive_vector("vp90-2-02-size-32x32.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_deltaq() {
    drive_vector("vp90-2-03-deltaq.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_resize() {
    drive_vector("vp90-2-05-resize.ivf");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_tile_1x2() {
    drive_vector("vp90-2-08-tile_1x2.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_tile_1x4() {
    drive_vector("vp90-2-08-tile_1x4.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_tile_1x8() {
    drive_vector("vp90-2-08-tile_1x8.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_aq2() {
    drive_vector("vp90-2-09-aq2.webm");
}

#[test]
#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
fn vec_show_existing_frame() {
    drive_vector("vp90-2-10-show-existing-frame.webm");
}
