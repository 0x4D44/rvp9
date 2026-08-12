//! G2 (decoder-hardening V6 §G2) — focused regression test for the
//! lossless decode path. Confirms that both frames of
//! `vp90-2-00-quantizer-00.webm` (the AOM conformance lossless vector)
//! decode bit-exact against the libvpx-published `.md5` digests.
//!
//! Before G2, both frames returned `DecodeError::Unimplemented` from
//! the keyframe / inter short-circuits at `lib.rs:735` / `lib.rs:1114`,
//! and the umbrella conformance harness counted them as skips. This
//! test pins the success path explicitly so a future regression in the
//! WHT kernel or its dispatch wiring fails loudly here even if the
//! umbrella suite's `MIN_PASSES` is left un-ratcheted by an inattentive
//! merge.
//!
//! The lossless vector is small — 2 frames at 352×288 — so running it
//! directly as a unit test is cheap. The WHT kernel itself has
//! algebraic unit tests next to its implementation
//! (`crates/rvp9-core/src/transforms/inverse.rs::tests::iwht*`).

use rvp9_decoder::{DecodeError, Decoder};
use rvp9_harness::container::read_packets;
use rvp9_harness::md5file::read_md5_file;
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
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

/// Decode every frame of the lossless conformance vector and assert
/// every shown frame's reconstructed planes hash to the libvpx-
/// published digest. No skips, no mismatches: the WHT path must close
/// both keyframe (`lib.rs:734`) and inter (`lib.rs:1113`) sites.
#[test]
fn lossless_quantizer_00_decodes_bit_exact() {
    let vector = vectors_dir().join("vp90-2-00-quantizer-00.webm");
    let md5_path = vectors_dir().join("vp90-2-00-quantizer-00.webm.md5");
    if !vector.exists() {
        eprintln!(
            "skip: {} not present (run scripts/fetch-vectors.sh)",
            vector.display()
        );
        return;
    }

    let (data, packets) = read_packets(&vector).expect("container parse");
    let md5_entries = read_md5_file(&md5_path).expect(".md5 parse");

    // Pin to one worker (G5 review M2 — avoid thread amplification).
    let mut decoder = Decoder::with_threads(NonZeroUsize::new(1).expect("1 is non-zero"));
    let mut shown_idx: usize = 0;
    let mut passed = 0usize;
    let mut mismatched = 0usize;
    let mut skipped = 0usize;
    let mut lossless_frames_decoded = 0usize;

    for range in &packets {
        let parsed = decoder
            .parse_packet(range.slice(&data))
            .expect("packet header parse");
        for frame in parsed {
            // Pre-check: the tail must signal lossless on every frame
            // in this vector (cq-level=0). If not, the vector's not the
            // one we expected — fail loudly rather than silently pass.
            if let Some(tail) = frame.header.tail.as_ref() {
                assert!(
                    tail.quantization.is_lossless(),
                    "vp90-2-00-quantizer-00 frame should be lossless; \
                     header reports base_qindex={}",
                    tail.quantization.base_qindex
                );
                lossless_frames_decoded += 1;
            }

            let shown = frame.header.show_frame
                || matches!(
                    frame.header.kind,
                    rvp9_core::bitstream::FrameKind::ShowExistingFrame { .. }
                );
            match decoder.decode_frame(&frame) {
                Ok(buf) => {
                    if !shown {
                        continue;
                    }
                    let got_hex = hex16(md5_of_yv12_visible(&buf));
                    let want = md5_entries
                        .get(shown_idx)
                        .expect("md5 entry for shown frame");
                    let want_hex = want
                        .md5_hex_visible()
                        .expect("md5 ledger entry should carry visible digest");
                    if want_hex == got_hex {
                        passed += 1;
                    } else {
                        mismatched += 1;
                        eprintln!(
                            "lossless mismatch @ shown={shown_idx}: \
                             got {got_hex}, want {want_hex}"
                        );
                    }
                    shown_idx += 1;
                }
                Err(DecodeError::Unimplemented { reason }) => {
                    // G2 must close every lossless short-circuit. Any
                    // surviving `Unimplemented` on this vector is a
                    // regression.
                    panic!("lossless decode hit Unimplemented (regression of G2 wiring): {reason}");
                }
                Err(other) => {
                    eprintln!("lossless decode error: {other:?}");
                    skipped += 1;
                    if shown {
                        shown_idx += 1;
                    }
                }
            }
        }
    }

    assert!(
        lossless_frames_decoded > 0,
        "expected at least one frame with lossless tail; got 0"
    );
    assert_eq!(
        mismatched, 0,
        "lossless vector should be bit-exact ({passed} passed, {mismatched} mismatched, {skipped} skipped)"
    );
    assert_eq!(
        skipped, 0,
        "lossless vector should have no skips ({passed} passed, {mismatched} mismatched, {skipped} skipped)"
    );
    assert!(
        passed >= 2,
        "expected at least 2 lossless frames to pass; got {passed}"
    );
}
