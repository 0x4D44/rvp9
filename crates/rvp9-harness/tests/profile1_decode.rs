//! VP9 **Profile 1** end-to-end decode conformance oracle.
//!
//! ## Why this test exists
//!
//! Profile 1 carries non-4:2:0 chroma subsampling. This milestone
//! (G7.1c) lands the *4:4:4* case — the simplest Profile-1 format,
//! where the chroma grid is identical to the luma grid. Getting it
//! byte-exact validates the whole `ss`-aware decode chain end to end:
//! the uncompressed-header chroma parse + inter-frame chroma
//! forwarding, the parametric MV / motion-compensation geometry
//! (G7.1a/b), intra prediction on full-resolution chroma, and the
//! `LF_PATH_444` loop-filter path (chroma filtered through the luma
//! `ss00` masks).
//!
//! ## The oracle
//!
//! Each vector under `tests/corpus/profile1/` ships with a sibling
//! `<file>.md5` ledger: one line per *shown* frame holding the MD5 of
//! the visible Y‖U‖V planes at native chroma dims, in the harness
//! format `<hex>  <stem>-WxH-NNNN.i444`. Those ledgers were verified
//! byte-exact against libvpx. We decode each vector, hash every shown
//! frame with [`md5_of_yv12_visible`] (which already handles arbitrary
//! chroma dims), and assert zero mismatches AND that the shown-frame
//! count equals the ledger length (so a truncated decode can't pass
//! vacuously).
//!
//! ## No silent skip
//!
//! Like `frame_parallel_regression.rs`, this test PANICS if a vector or
//! its ledger is missing — the coverage must be impossible to lose.
//!
//! ## Adding the next milestone's vectors
//!
//! When 4:2:2 / 4:4:0 land, add their `(vector, ledger)` stems to
//! [`PROFILE1_VECTORS`] — that's the only change needed here.

use rvp9_core::bitstream::FrameKind;
use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use rvp9_harness::md5file::read_md5_file;
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
use std::path::PathBuf;

/// The Profile-1 conformance vectors gated by this milestone. Each
/// entry is the bare filename under `tests/corpus/profile1/`; the
/// ledger is the same name with `.md5` appended.
///
/// G7.1c gates the two 4:4:4 vectors. Adding the 4:2:2 / 4:4:0 vectors
/// next milestone is a one-line append here.
const PROFILE1_VECTORS: &[&str] = &[
    "vp91-2-04-yuv444.webm", // official WebM-project 4:4:4 vector
    "foreman_c444.ivf",      // self-encoded 4:4:4 (vpxenc default knobs)
];

/// Resolve `tests/corpus/profile1/<name>` from the workspace root.
/// `CARGO_MANIFEST_DIR` = `crates/rvp9-harness`; the root is two levels
/// up (same idiom as `frame_parallel_regression.rs`).
fn profile1_vector(name: &str) -> PathBuf {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("tests")
        .join("corpus")
        .join("profile1")
        .join(name)
}

/// The sibling ledger path: the full vector filename with `.md5`
/// appended (so it works for both `.webm` and `.ivf` vectors).
fn ledger_path(vector: &PathBuf) -> PathBuf {
    let mut s = vector.clone().into_os_string();
    s.push(".md5");
    PathBuf::from(s)
}

/// Decode one Profile-1 vector and gate every shown frame's visible
/// MD5 against the sibling ledger. Returns `(shown_frames, mismatches)`
/// after asserting the zero-mismatch + full-coverage invariants.
fn decode_and_gate(name: &str) -> (usize, usize) {
    let vector = profile1_vector(name);
    let ledger = ledger_path(&vector);

    // Fail loudly — never skip. (M1 lesson: missing coverage that
    // silently evaporates is worse than no coverage.)
    assert!(
        vector.exists(),
        "Profile-1 vector missing: {}",
        vector.display(),
    );
    assert!(
        ledger.exists(),
        "Profile-1 ledger missing: {}",
        ledger.display(),
    );

    let (data, packets) = read_packets(&vector)
        .unwrap_or_else(|e| panic!("container parse {}: {e}", vector.display()));
    let entries = read_md5_file(&ledger)
        .unwrap_or_else(|e| panic!("ledger parse {}: {e}", ledger.display()));

    // Pin to one worker — deterministic, no thread amplification.
    let mut decoder = Decoder::with_threads(NonZeroUsize::new(1).expect("1 is non-zero"));
    let mut shown_idx: usize = 0;
    let mut mismatched: usize = 0;

    for range in &packets {
        let parsed = decoder
            .parse_packet(range.slice(&data))
            .unwrap_or_else(|e| panic!("parse_packet {}: {e}", vector.display()));
        for frame in parsed {
            let shown = frame.header.show_frame
                || matches!(frame.header.kind, FrameKind::ShowExistingFrame { .. });
            let buf = decoder.decode_frame(&frame).unwrap_or_else(|e| {
                panic!(
                    "{}: decode error on shown-frame {shown_idx} (kind={:?}): {e}",
                    vector.display(),
                    frame.header.kind
                )
            });
            if !shown {
                continue;
            }
            let got_hex = hex16(md5_of_yv12_visible(&buf));
            match entries.get(shown_idx) {
                Some(e) if e.md5_hex_visible().as_deref() == Some(got_hex.as_str()) => {}
                Some(e) => {
                    mismatched += 1;
                    println!(
                        "  {name} shown-frame {shown_idx}: got {got_hex}, want {}",
                        e.md5_hex_visible()
                            .as_deref()
                            .unwrap_or("<no-visible-digest>"),
                    );
                }
                None => panic!(
                    "{name}: decoded more shown frames than the ledger has ({} \
                     entries) — shown_idx {shown_idx} out of range",
                    entries.len()
                ),
            }
            shown_idx += 1;
        }
    }

    println!(
        "profile1_decode[{name}]: {shown_idx} shown frame(s) compared against \
         {} ledger entries, {mismatched} mismatch(es)",
        entries.len()
    );

    // Zero-mismatch gate — the load-bearing assertion.
    assert_eq!(
        mismatched, 0,
        "{name}: Profile-1 visible MD5 mismatches: {mismatched}",
    );
    // A truncated decode must not pass vacuously.
    assert_eq!(
        shown_idx,
        entries.len(),
        "{name}: shown-frame count ({shown_idx}) does not match ledger length \
         ({}) — decode truncated or fixture/ledger out of sync",
        entries.len()
    );

    (shown_idx, mismatched)
}

#[test]
fn profile1_4x4x4_vectors_decode_bit_exact() {
    let mut total_frames = 0usize;
    for name in PROFILE1_VECTORS {
        let (frames, mismatched) = decode_and_gate(name);
        assert_eq!(mismatched, 0, "{name} had mismatches");
        total_frames += frames;
    }
    assert!(
        total_frames > 0,
        "no shown frames decoded across the Profile-1 corpus",
    );
}
