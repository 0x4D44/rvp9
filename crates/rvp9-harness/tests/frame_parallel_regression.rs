//! Frame-parallel (`frame_parallel_decoding_mode == 1`) regression oracle.
//!
//! ## Why this test exists
//!
//! Wave M (commit 8bd8f01) fixed a missing `!frame_parallel_decoding_mode`
//! term in the two backward-adaptation gates in
//! `crates/rvp9-decoder/src/lib.rs` (the keyframe coef-adapt gate and the
//! inter mode/coef/mv-adapt gate). With the term missing, the decoder ran
//! backward adaptation on frames where libvpx does NOT — corrupting the
//! frame-context banks carried into subsequent frames and diverging from the
//! reference byte-for-byte.
//!
//! That bug evaded ALL 201 official WebM-project conformance vectors (the M1
//! lesson): every official vector is encoded with
//! `frame_parallel_decoding_mode == 0`, so the missing term was a no-op on
//! them. A *default* `vpxenc` encode, however, sets
//! `frame_parallel_decoding_mode == 1` — which is the common real-world
//! case. The official corpus is therefore blind to this entire class of bug.
//!
//! This test closes that blind spot with a frozen, in-repo fixture
//! (`tests/corpus/fixtures/foreman-fp1.ivf`) produced by a DEFAULT vpxenc
//! encode (`--passes=1 --lag-in-frames=25`, no `--auto-alt-ref=0`), which
//! carries `frame_parallel_decoding_mode == 1`. The visible-plane MD5 of
//! every shown frame is gated bit-exact against the sibling libvpx-generated
//! ledger.
//!
//! ## No silent skip
//!
//! Unlike `real_world.rs` (which skips cleanly when no fixtures are present),
//! this test PANICS if the fixture or its ledger is missing. The whole point
//! is that this coverage cannot silently evaporate. Regenerate the fixture
//! with:
//!
//! ```text
//! pwsh scripts/encode-corpus.ps1 -Src tests/corpus/foreman_cif.y4m \
//!     -Out tests/corpus/fixtures/foreman-fp1.ivf -Limit 30 \
//!     -ExtraArgs '--passes=1','--lag-in-frames=25' -WithLedger
//! ```

use rvp9_core::bitstream::FrameKind;
use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use rvp9_harness::md5file::read_triangulated_ledger;
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
use std::path::PathBuf;

/// Resolve `tests/corpus/fixtures/foreman-fp1.ivf` from the workspace root.
/// `CARGO_MANIFEST_DIR` = `crates/rvp9-harness`; the root is two levels up
/// (same idiom as `real_world.rs`).
fn fixture_ivf() -> PathBuf {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("tests")
        .join("corpus")
        .join("fixtures")
        .join("foreman-fp1.ivf")
}

const REGEN_HINT: &str = "Regenerate it with:\n  \
    pwsh scripts/encode-corpus.ps1 -Src tests/corpus/foreman_cif.y4m \
    -Out tests/corpus/fixtures/foreman-fp1.ivf -Limit 30 \
    -ExtraArgs '--passes=1','--lag-in-frames=25' -WithLedger";

#[test]
fn frame_parallel_corpus_decodes_bit_exact() {
    let ivf = fixture_ivf();

    // Fail loudly — never skip. Missing coverage must be impossible to miss.
    assert!(
        ivf.exists(),
        "frame-parallel regression fixture missing: {}\n{REGEN_HINT}",
        ivf.display(),
    );

    let (data, packets) =
        read_packets(&ivf).unwrap_or_else(|e| panic!("container parse {}: {e}", ivf.display()));
    // Wave O: triangulated oracle. Requires both <ivf>.md5 (vpxdec) and
    // <ivf>.ffmpeg.md5 to be present AND to agree on every frame's hash
    // before this returns. If they disagree, the panic message points to
    // libvpx/ffmpeg as the suspect — never the consumer's decoder.
    let entries = read_triangulated_ledger(&ivf);

    // Pin to one worker (mirrors real_world.rs — avoid thread amplification).
    let mut decoder = Decoder::with_threads(NonZeroUsize::new(1).expect("1 is non-zero"));
    let mut shown_idx: usize = 0;
    let mut mismatched: usize = 0;

    for range in &packets {
        let parsed = decoder
            .parse_packet(range.slice(&data))
            .unwrap_or_else(|e| panic!("parse_packet {}: {e}", ivf.display()));
        for frame in parsed {
            let shown = frame.header.show_frame
                || matches!(frame.header.kind, FrameKind::ShowExistingFrame { .. });
            let buf = decoder.decode_frame(&frame).unwrap_or_else(|e| {
                panic!(
                    "{}: decode error on shown-frame {shown_idx} (kind={:?}): {e}",
                    ivf.display(),
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
                        "  shown-frame {shown_idx}: got {got_hex}, want {}",
                        e.md5_hex_visible().as_deref().unwrap_or("<no-visible-digest>"),
                    );
                }
                None => panic!(
                    "decoded more shown frames than the ledger has ({} entries) — \
                     shown_idx {shown_idx} out of range",
                    entries.len()
                ),
            }
            shown_idx += 1;
        }
    }

    println!(
        "frame_parallel_regression: {shown_idx} shown frame(s) compared against \
         {} ledger entries, {mismatched} mismatch(es)",
        entries.len()
    );

    // Zero-mismatch gate — the load-bearing assertion.
    assert_eq!(
        mismatched, 0,
        "frame-parallel visible MD5 mismatches: {mismatched} (frame_parallel=1 \
         backward-adaptation regression — see commit 8bd8f01)"
    );

    // A truncated decode must not pass vacuously: every ledger entry must
    // have been compared against a decoded shown frame.
    assert_eq!(
        shown_idx,
        entries.len(),
        "shown-frame count ({shown_idx}) does not match ledger length ({}) — \
         decode was truncated or the fixture/ledger are out of sync",
        entries.len()
    );
}
