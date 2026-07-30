//! VP9 **Profile 1** end-to-end decode conformance oracle.
//!
//! ## Why this test exists
//!
//! Profile 1 carries non-4:2:0 chroma subsampling. The gate now covers
//! **all three** Profile-1 chroma layouts end to end:
//!
//! - **4:4:4** (G7.1c milestone) — chroma grid == luma grid; the
//!   simplest case. Validates the `ss`-aware decode chain
//!   (parametric geometry G7.1a/b, full-res intra-chroma, and the
//!   `LF_PATH_444` loop-filter path with `ss00` masks).
//! - **4:2:2** (G7.2 milestone) — asymmetric (half horizontal chroma,
//!   full vertical). Exercises the `LF_PATH_SLOW` loop-filter path
//!   (`filter_block_plane_non420`) plus the ss-aware MC / residual.
//! - **4:4:0** (G7.2 milestone) — asymmetric (full horizontal chroma,
//!   half vertical). Same `LF_PATH_SLOW` path as 4:2:2.
//!
//! Profile 1 of course also subsumes 4:2:0, but that's already covered
//! by the main `conformance` test (Profile-0 vectors).
//!
//! ## The oracle
//!
//! Each vector under `tests/corpus/profile1/` ships with a sibling
//! `<file>.md5` ledger: one line per *shown* frame holding the MD5 of
//! the visible Y‖U‖V planes at native chroma dims, in the harness
//! format `<hex>  <stem>-WxH-NNNN.<i420|i422|i440|i444>`. Those ledgers
//! were verified byte-exact against libvpx. We decode each vector,
//! hash every shown frame with [`md5_of_yv12_visible`] (which already
//! handles arbitrary chroma dims), and assert zero mismatches AND that
//! the shown-frame count equals the ledger length (so a truncated
//! decode can't pass vacuously).
//!
//! ## No silent skip
//!
//! Like `frame_parallel_regression.rs`, this test PANICS if a vector or
//! its ledger is missing — the coverage must be impossible to lose.

use rvp9_core::bitstream::FrameKind;
use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use rvp9_harness::md5file::read_triangulated_ledger;
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
use std::path::PathBuf;

/// The Profile-1 conformance vectors gated by this milestone. Each
/// entry is the bare filename under `tests/corpus/profile1/`; the
/// ledger is the same name with `.md5` appended.
///
/// G7.2 gates the official WebM-project vector AND a self-encoded
/// `vpxenc`-default vector for each of 4:4:4, 4:2:2, and 4:4:0.
///
/// `foreman_c440.ivf` was previously gated out behind a Wave N
/// `#[ignore]`d follow-up because frame 0 cascaded all 12 frames. The
/// root cause (Wave P, 2026-05-23) was `reset_skip_context` covering
/// only the sub-block chroma extent for sub-8 blocks instead of the
/// parent-8x8 mi cell extent (1 cell instead of 2 for 4:4:0 chroma).
/// That left a stale `1` in above-context cells beyond the sub-block,
/// which mis-routed the entropy coder several blocks later. Fix is in
/// `crates/rvp9-decoder/src/tile.rs::reset_skip_context`.
const PROFILE1_VECTORS: &[&str] = &[
    // 4:4:4 (G7.1c milestone)
    "vp91-2-04-yuv444.webm",
    "foreman_c444.ivf",
    "foreman_c444_aq2.ivf",
    // 4:2:2 (G7.2 milestone) — asymmetric, half horizontal chroma.
    "vp91-2-04-yuv422.webm",
    "foreman_c422.ivf",
    // 4:4:0 (G7.2 milestone) — asymmetric, half vertical chroma.
    "vp91-2-04-yuv440.webm",
    "foreman_c440.ivf", // Wave P fix; previously the parser-desync follow-up.
    // Track T-2: sharpness=7 triple — exercises LF sharpness threshold ×
    // each chroma subsampling, gating `filter_block_plane_non420` and
    // `LF_PATH_444` under elevated sharpness.
    "foreman_c444_sharp7.ivf",
    "foreman_c422_sharp7.ivf",
    "foreman_c440_sharp7.ivf",
    // Track T-2: 2-pass triple — exercises different bit-allocation
    // decisions (alt-ref frames, lag) across all three Profile-1 chroma
    // formats. Covers inter-prediction patterns absent from the
    // single-pass default vectors.
    "foreman_c444_2pass.ivf",
    "foreman_c422_2pass.ivf",
    "foreman_c440_2pass.ivf",
    // Track T-2: 720p variants — large-frame chroma at ≥720p.
    // Catches per-row mask-packing bugs at high mi_rows and exercises
    // multi-tile (2 tile cols) + tile-aware chroma loop-filter.
    "testsrc_c444_720p.ivf",
    "testsrc_c444_720p_aq2.ivf",
    "testsrc_c422_720p_tile2.ivf", // tile_columns=1 (2 tiles)
    "testsrc_c440_720p.ivf",       // 4:4:0 large-frame
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

/// Decode one Profile-1 vector and gate every shown frame's visible
/// MD5 against the **triangulated** ledger (vpxdec + ffmpeg must agree
/// first; only then is our decoder compared). Returns `(shown_frames,
/// mismatches)` after asserting the zero-mismatch + full-coverage
/// invariants.
fn decode_and_gate_with_threads(name: &str, threads: usize) -> (usize, usize) {
    let vector = profile1_vector(name);

    // Fail loudly — never skip. (M1 lesson: missing coverage that
    // silently evaporates is worse than no coverage.)
    assert!(
        vector.exists(),
        "Profile-1 vector missing: {}",
        vector.display(),
    );

    let (data, packets) = read_packets(&vector)
        .unwrap_or_else(|e| panic!("container parse {}: {e}", vector.display()));
    // Wave O: triangulated oracle — `read_triangulated_ledger` loads
    // both <vector>.md5 (vpxdec) and <vector>.ffmpeg.md5 and asserts
    // they agree on every shown-frame's hash BEFORE this returns.
    // If the two oracles disagree the panic message names libvpx /
    // ffmpeg as the suspect, NOT our decoder.
    let entries = read_triangulated_ledger(&vector);

    let mut decoder =
        Decoder::with_threads(NonZeroUsize::new(threads).expect("thread count is non-zero"));
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

fn decode_and_gate(name: &str) -> (usize, usize) {
    decode_and_gate_with_threads(name, 1)
}

#[test]
fn profile1_vectors_decode_bit_exact() {
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

#[test]
fn profile1_tile2_parallel_scratch_decodes_bit_exact() {
    let jobs_before =
        rvp9_decoder::RVP9_G5_1_PARALLEL_JOBS.load(std::sync::atomic::Ordering::Relaxed);
    let (frames, mismatched) = decode_and_gate_with_threads("testsrc_c422_720p_tile2.ivf", 2);
    let jobs_after =
        rvp9_decoder::RVP9_G5_1_PARALLEL_JOBS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(frames > 0, "parallel tile fixture produced no shown frames");
    assert_eq!(mismatched, 0, "parallel scratch decode diverged");
    assert!(
        jobs_after.wrapping_sub(jobs_before) >= 2 * frames as u32,
        "two-tile fixture did not exercise parallel worker dispatch"
    );
}

/// Wave P fix-lock test for the foreman_c440 parser desync. The vector
/// is also in `PROFILE1_VECTORS` (so it's gated by the main test), but
/// this single-vector test stays as a named regression marker pointing
/// back to the journal — and so a future regression on this vector
/// produces a focused, named failure rather than just "one of six".
///
/// Root cause was `reset_skip_context` covering the sub-block chroma
/// extent rather than the parent-8x8 mi cell extent for sub-8 blocks
/// (`crates/rvp9-decoder/src/tile.rs::reset_skip_context`). See
/// `wrk_journals/2026.05.23 - JRN - Wave P foreman_c440 parser desync fix.md`.
#[test]
fn profile1_440_foreman_decodes_bit_exact_wave_p() {
    let (_frames, mismatched) = decode_and_gate("foreman_c440.ivf");
    assert_eq!(mismatched, 0, "foreman_c440.ivf Wave P fix-lock");
}

/// SIG4 fix-lock (2026.06.04 longclip campaign). The G7.2 commit `a278399`
/// updated `decode_tile_intra`'s chroma allow-list to all four formats but
/// left `decode_tile_inter`'s `debug_assert!` at `Yuv420 | Yuv444`, wrongly
/// tripping on legal 4:2:2 / 4:4:0 INTER frames in debug builds. That silently
/// broke this whole test (its `foreman_c422*`/`foreman_c440*`/`vp91-2-04-yuv422`/
/// `vp91-2-04-yuv440` vectors are all `I P P P…` inter streams) from 2026-05-23
/// until the fix relaxed the inter assert to mirror the intra one. The recon
/// path was always ss-aware (release builds — assert compiled out — decoded
/// these bit-exact); only the debug guard was wrong. This focused marker
/// decodes one 4:2:2 and one 4:4:0 inter stream bit-exact so a regression on the
/// inter asymmetric-chroma path produces a named failure rather than "one of
/// fifteen". See `wrk_docs/2026.06.04 - HLD - SIG4 profile-1 asymmetric-chroma
/// inter decode assert.md`.
#[test]
fn profile1_422_440_inter_decodes_bit_exact_sig4() {
    for name in ["foreman_c422.ivf", "foreman_c440.ivf"] {
        let (frames, mismatched) = decode_and_gate(name);
        assert_eq!(
            mismatched, 0,
            "{name}: SIG4 inter asymmetric-chroma fix-lock"
        );
        assert!(
            frames >= 2,
            "{name}: expected an inter stream (>=2 shown frames)"
        );
    }
}

/// Non-asserting variant for probing vectors that are expected to fail
/// (rvp9 real-bug follow-ups). Returns `(shown_frames, mismatches)`.
fn decode_and_count(name: &str) -> (usize, usize) {
    let vector = profile1_vector(name);
    assert!(
        vector.exists(),
        "Profile-1 vector missing: {}",
        vector.display()
    );

    let (data, packets) = read_packets(&vector)
        .unwrap_or_else(|e| panic!("container parse {}: {e}", vector.display()));
    let entries = read_triangulated_ledger(&vector);

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
            if let Some(e) = entries.get(shown_idx)
                && e.md5_hex_visible().as_deref() != Some(got_hex.as_str())
            {
                mismatched += 1;
            }
            shown_idx += 1;
        }
    }
    (shown_idx, mismatched)
}

/// SIG1/SIG2 end-to-end regression guard (was Track T-2 follow-up;
/// activated per KILN-00007). kfonly (`--kf-max-dist=1`) Profile-1
/// streams force every frame to be an independent keyframe, so they
/// exercise multi-keyframe independence that the default single-keyframe
/// I-P-P-P `PROFILE1_VECTORS` do not. These streams previously mismatched
/// from ~frame 3 onward (decoder state leaking across keyframes —
/// `FrameContext` banks and above/left coef context). The track2 harvest
/// fixed it: SIG1 (reset all `FrameContext` banks on every keyframe) +
/// SIG2 (above/left coef-context frame-edge clamp). They now decode
/// bit-exact vs the committed oracle MD5s and are the only end-to-end
/// guard against a SIG1 cross-keyframe-independence regression; vpxdec
/// and ffmpeg agree with these MD5s.
#[test]
fn profile1_444_kfonly_decodes_bit_exact() {
    let (shown, mismatched) = decode_and_count("foreman_c444_kfonly.ivf");
    assert_eq!(shown, 12);
    assert_eq!(mismatched, 0);
}

#[test]
fn profile1_422_kfonly_decodes_bit_exact() {
    let (shown, mismatched) = decode_and_count("foreman_c422_kfonly.ivf");
    assert_eq!(shown, 12);
    assert_eq!(mismatched, 0);
}

#[test]
fn profile1_440_kfonly_decodes_bit_exact() {
    let (shown, mismatched) = decode_and_count("foreman_c440_kfonly.ivf");
    assert_eq!(shown, 12);
    assert_eq!(mismatched, 0);
}

/// Cyclic-refresh segmentation regression: when `update_map == false`,
/// each covered previous-map cell must be copied forward unchanged.
/// Splatting the block's minimum inherited ID corrupts later frames whose
/// partitioning exposes a different subset of those cells.
#[test]
fn profile1_444_aq2_follow_up() {
    let (shown, mismatched) = decode_and_count("foreman_c444_aq2.ivf");
    assert_eq!(shown, 12);
    assert_eq!(mismatched, 0);
}

#[test]
fn profile1_720p_aq2_follow_up() {
    let (shown, mismatched) = decode_and_count("testsrc_c444_720p_aq2.ivf");
    assert_eq!(shown, 12);
    assert_eq!(mismatched, 0);
}
