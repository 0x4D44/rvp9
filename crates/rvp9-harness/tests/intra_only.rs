//! G1 — IntraOnly frame decode path.
//!
//! Decode `vp90-2-10-show-existing-frame.webm` end-to-end and assert
//! every shown frame matches the libvpx-reference MD5. This vector
//! mixes KeyFrame / IntraOnly / InterFrame / ShowExistingFrame kinds;
//! before G1 lands, 8 of its 13 shown frames are skipped because the
//! decoder returns `DecodeError::Unimplemented` for `FrameKind::IntraOnly`.
//!
//! After G1 closes the IntraOnly arm, all 13 shown frames must decode
//! bit-exact (no `Unimplemented`, no mismatch). This test is a
//! per-vector ratchet: it asserts EXACTLY the libvpx MD5 for every
//! shown frame and fails loudly on any deviation.
//!
//! Oracle: libvpx-anchored at scale (matches the conformance harness's
//! per-frame MD5 check against `vpxdec --rawvideo` output captured
//! in `vp90-2-10-show-existing-frame.webm.md5`).

use rvp9_decoder::{DecodeError, Decoder};
use rvp9_harness::container::read_packets;
use rvp9_harness::md5file::read_md5_file;
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
use std::path::PathBuf;

fn vector_path() -> PathBuf {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("tests")
        .join("conformance")
        .join("vectors")
        .join("vp90-2-10-show-existing-frame.webm")
}

/// G1 acceptance test: every shown frame in `vp90-2-10-show-existing-frame.webm`
/// matches its libvpx-reference MD5. Before G1: 5 pass, 8 skipped
/// (IntraOnly `Unimplemented`). After G1: 13 pass, 0 skipped.
#[test]
fn show_existing_frame_all_thirteen_shown_frames_match_reference() {
    let vector = vector_path();
    if !vector.exists() {
        eprintln!(
            "vector {} not present — run scripts/fetch-vectors.sh",
            vector.display()
        );
        return;
    }
    let md5_path = vector.with_extension("webm.md5");
    let (data, packets) = read_packets(&vector).expect("container parse");
    let md5_entries = read_md5_file(&md5_path).expect(".md5 parse");
    assert_eq!(
        md5_entries.len(),
        13,
        "show-existing-frame.webm has 13 shown frames per its .md5"
    );

    // Pin to one worker (G5 review M2 — avoid thread amplification).
    let mut decoder = Decoder::with_threads(NonZeroUsize::new(1).expect("1 is non-zero"));
    let mut shown_idx: usize = 0;
    let mut matched: Vec<bool> = vec![false; md5_entries.len()];
    let mut failures: Vec<String> = Vec::new();
    for range in &packets {
        let parsed = decoder
            .parse_packet(range.slice(&data))
            .expect("header parse");
        for frame in parsed {
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
                    let want = md5_entries.get(shown_idx);
                    match want {
                        Some(e) if e.md5_hex_visible().as_deref() == Some(got_hex.as_str()) => {
                            matched[shown_idx] = true;
                        }
                        Some(e) => {
                            let want_hex = e
                                .md5_hex_visible()
                                .unwrap_or_else(|| "<no-visible-digest>".to_string());
                            failures.push(format!(
                                "frame[{shown_idx}] kind={:?}: md5 mismatch — got {got_hex} want {want_hex}",
                                frame.header.kind
                            ));
                        }
                        None => {
                            failures.push(format!("frame[{shown_idx}]: no reference md5 entry"));
                        }
                    }
                    shown_idx += 1;
                }
                Err(DecodeError::Unimplemented { reason }) => {
                    // G1 acceptance hard gate: after G1's IntraOnly +
                    // ShowExistingFrame fixes, no frame in
                    // `vp90-2-10-show-existing-frame.webm` should hit
                    // `Unimplemented`. Treating it as recoverable here
                    // would repeat the conformance-harness skip-folding
                    // bug (K-5). Hard-fail immediately so a regression
                    // surfaces with the exact offending frame, instead
                    // of being summarised in the failures vector at end.
                    panic!(
                        "G1 regression: frame[{shown_idx}] kind={:?} returned \
                         Unimplemented (reason: {reason}). Post-G1 the IntraOnly \
                         path must decode every shown frame in show-existing-frame.webm.",
                        frame.header.kind,
                    );
                }
                Err(e) => {
                    if shown {
                        failures.push(format!(
                            "frame[{shown_idx}] kind={:?}: decode error {e:?}",
                            frame.header.kind
                        ));
                        shown_idx += 1;
                    }
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "G1 IntraOnly decode regressions ({} failure(s)):\n  {}",
        failures.len(),
        failures.join("\n  "),
    );
    let total_matched = matched.iter().filter(|&&m| m).count();
    assert_eq!(
        total_matched,
        md5_entries.len(),
        "expected all {} shown frames to match libvpx MD5 reference, only {total_matched} did",
        md5_entries.len(),
    );
}
