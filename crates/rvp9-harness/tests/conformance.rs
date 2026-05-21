//! End-to-end test: drives every conformance vector through our
//! decoder and asserts the harness does not produce any *mismatches*.
//!
//! Skipped frames are expected (and counted) until the full decode
//! path lands; they do NOT fail the test. Mismatches DO fail the
//! test — that's the load-bearing gate.
//!
//! Vector / `.md5` parse failures (i.e. broken harness plumbing)
//! also fail the test.

use rvp9_decoder::{DecodeError, Decoder};
use rvp9_harness::container::read_packets;
use rvp9_harness::md5file::read_md5_file;
use rvp9_harness::{hex16, md5_of_yv12_visible};
use std::num::NonZeroUsize;
use std::path::PathBuf;

/// Expected sub-frame count for a transport packet, derived purely
/// from the trailing superframe-index byte (VP9 spec §B.3) via an
/// **inline re-parse** that does NOT call into `rvp9_core`.
///
/// G8.0 review (2026-05-21): the original implementation delegated to
/// `parse_superframe_index` — the SAME function `Decoder::parse_packet`
/// already calls — so the oracle was a tautology (any bug in the
/// superframe parser would corrupt both sides equally and the oracle
/// would stay silent). This re-implementation is genuinely independent:
/// it reads only the trailing marker byte and decodes the 3 low bits
/// directly, so a regression that drops a sub-frame mid-loop inside
/// `parse_packet` (or anywhere downstream) lights up immediately.
///
/// Wire format (VP9 spec §B.3 / libvpx `vp9_parse_superframe_index`):
///   - Last byte's top 3 bits == `0b110` (`marker & 0xe0 == 0xc0`) ⇒
///     superframe; count = `(marker & 0b0000_0111) + 1` (1..=8).
///   - Otherwise ⇒ single-frame packet, count = 1.
///
/// Truncated/malformed indices fall through to count = 1: the header-
/// parse arm below already reports them via the `Err` path, so we
/// don't want to double-fault the same packet.
fn expected_sub_frame_count(packet: &[u8]) -> usize {
    let Some(&marker) = packet.last() else {
        return 1;
    };
    if (marker & 0xe0) != 0xc0 {
        return 1;
    }
    ((marker & 0x07) as usize) + 1
}

fn vectors_dir() -> PathBuf {
    // Find the workspace root by walking up from this file's
    // location. `CARGO_MANIFEST_DIR` is the crate root
    // (`crates/rvp9-harness`); the workspace root is two levels up.
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("tests")
        .join("conformance")
        .join("vectors")
}

#[test]
fn conformance_runner_walks_all_vectors() {
    let dir = vectors_dir();
    if !dir.exists() {
        eprintln!(
            "vectors dir {} not present — run scripts/fetch-vectors.ps1",
            dir.display()
        );
        return;
    }

    let mut vectors: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|x| matches!(x, "ivf" | "webm"))
        })
        .collect();
    vectors.sort();

    assert!(
        !vectors.is_empty(),
        "no .ivf/.webm vectors found in {}",
        dir.display()
    );

    let mut total_passed = 0usize;
    let mut total_mismatched = 0usize;
    let mut total_skipped = 0usize;
    let mut total_expected = 0usize;
    // K-5 (HLD V6 §G1, lines 1806-1821): track non-`Unimplemented`
    // decode errors separately so the silent skip-folding bug —
    // historically both `Err` arms incremented `total_skipped`,
    // masking real decode failures behind the "expected" skip count
    // — cannot recur. Asserted to be zero after the loop.
    let mut total_other_err: usize = 0;
    // G7.0 review (2026-05-21): bucket `DecodeError::UnsupportedProfile`
    // separately from `Unimplemented`. Gated features (currently:
    // Profile 1 chroma via the G7.0 gate) are explicitly OK — they have
    // a named follow-up task. We do NOT assert `total_gated == 0`; the
    // supervisor expects this counter to climb during transitional
    // periods (e.g. when a Profile 1 vector enters the corpus before
    // G7.1 lands). Folding the gated-feature bucket back into
    // `total_skipped` would resurrect exactly the K-5 silent-skip
    // failure mode (a future Profile 1 vector classified as a benign
    // skip rather than a flagged gating event).
    let mut total_gated: usize = 0;

    for vector in &vectors {
        let md5_path = vector.with_extension(format!(
            "{}.md5",
            vector.extension().and_then(|s| s.to_str()).unwrap_or("")
        ));
        let (data, packets) = read_packets(vector).expect("container parse");
        let md5_entries = read_md5_file(&md5_path).expect(".md5 parse");
        total_expected += md5_entries.len();

        // Per-vector tally — only printed under
        // `RVP9_CONFORMANCE_PER_VECTOR=1` (off by default to keep the
        // CI log small).
        let report_per_vector = std::env::var("RVP9_CONFORMANCE_PER_VECTOR").is_ok();
        let mut v_passed = 0usize;
        let mut v_mismatched = 0usize;
        let mut v_skipped = 0usize;

        // Pin to one worker thread per G5 review (M2) — this loop
        // constructs a fresh `Decoder` per vector (×201 vectors), and
        // `Decoder::new()` would spawn `available_parallelism()`
        // threads each time. The conformance pipeline runs every tile
        // inline in the first-cut decoder, so additional workers sit
        // idle anyway.
        let mut decoder = Decoder::with_threads(NonZeroUsize::new(1).expect("1 is non-zero"));
        let mut shown_idx: usize = 0;
        for (packet_idx, range) in packets.iter().enumerate() {
            let packet_bytes = range.slice(&data);
            // G8 sub-frame-count sanity oracle (HLD V6 §G8 lines
            // 1489-1502): count the number of `ParsedFrame`s
            // `parse_packet` produces and compare to the count derived
            // independently from the trailing superframe-index byte
            // (VP9 spec §B.3). Any mismatch indicates a hidden-frame
            // ordering/count regression that the visible-only MD5
            // check cannot detect. Hard-fail with a diagnostic naming
            // the offending packet.
            let expected_sub_frames = expected_sub_frame_count(packet_bytes);
            let parsed = match decoder.parse_packet(packet_bytes) {
                Ok(p) => p,
                Err(_) => {
                    // header parse fail → treat the packet as a single
                    // skipped shown-frame.
                    total_skipped += 1;
                    shown_idx += 1;
                    continue;
                }
            };
            let decoded_sub_frames = parsed.frames().len();
            assert_eq!(
                decoded_sub_frames,
                expected_sub_frames,
                "G8 sub-frame-count sanity oracle: vector {} packet {} \
                 decoded {} sub-frame(s); superframe-index byte says \
                 {}. Hidden-frame ordering/count regression.",
                vector.file_name().unwrap_or_default().to_string_lossy(),
                packet_idx,
                decoded_sub_frames,
                expected_sub_frames,
            );
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
                                total_passed += 1;
                                v_passed += 1;
                            }
                            _ => {
                                total_mismatched += 1;
                                v_mismatched += 1;
                            }
                        }
                        shown_idx += 1;
                    }
                    Err(DecodeError::Unimplemented { .. }) => {
                        if shown {
                            total_skipped += 1;
                            v_skipped += 1;
                            shown_idx += 1;
                        }
                    }
                    Err(DecodeError::UnsupportedProfile { .. }) => {
                        // G7.0 review (2026-05-21): gated-feature
                        // bucket. The decoder DETECTED a known-
                        // unsupported feature (today: Profile 1 chroma
                        // via the G7.0 gate) and rejected it with a
                        // dedicated variant. Counted independently
                        // from `total_skipped` so the K-5 silent skip-
                        // folding failure mode cannot recur for gated
                        // features. Still advance `shown_idx` so
                        // subsequent shown frames line up against the
                        // .md5 list.
                        if shown {
                            total_gated += 1;
                            shown_idx += 1;
                        }
                    }
                    Err(_) => {
                        // K-5: any non-`Unimplemented` decode error is a
                        // real failure, not an expected skip. Count it
                        // separately so the post-loop assertion can hard-
                        // fail. Still advance `shown_idx` so subsequent
                        // shown frames line up against the .md5 list.
                        if shown {
                            total_other_err += 1;
                            shown_idx += 1;
                        }
                    }
                }
            }
        }
        if report_per_vector {
            println!(
                "  {}: {v_passed} passed / {v_mismatched} mismatched / {v_skipped} skipped",
                vector.file_name().unwrap_or_default().to_string_lossy(),
            );
        }
    }

    println!(
        "conformance: {total_passed} passed / {total_mismatched} mismatched / \
         {total_skipped} skipped / {total_gated} gated / {total_expected} total"
    );

    // Stage 3 EXIT — `total_mismatched == 0` is the final load-bearing
    // gate. With D.25's partial-bottom edge-clamp fix the last 16 mis-
    // matches lit up; we now assert hard zero instead of ratcheting.
    // `MIN_PASSES` is kept as defense-in-depth so any future regression
    // that drops `total_passed` (e.g. someone introduces an Unimplemented
    // path) still trips the gate.
    // Wave A R1 (2026-05-21): ratcheted 191 → 201 after G1 (frame_pool
    // sync) + G2 (WHT lossless) closed the last 10 skips. Full 14-vector
    // AOM corpus now passes bit-exact. Per HLD V6 §4 Wave A exit gate #3.
    const MIN_PASSES: usize = 201;
    // D.24 (2026-05-19): fix scaled-MC for resize vectors. Two
    // distinct bugs (one Y, one chroma) that surface together on any
    // inter frame whose reference frame has a different size than the
    // current frame:
    //
    //   1. `ScaleFactors` was built with `ref_buf.aligned_y_width()`
    //      and `inputs.cur_y_width` (the 8-aligned dims). Libvpx
    //      passes `y_crop_width` / `cm->width` — the *un-aligned*
    //      visible dims — to `vp9_setup_scale_factors_for_frame`
    //      (`vp9_decodeframe.c:2928-2935`). For 282×173 this differs
    //      by 6/3 pixels respectively; the Q14 ratio changes and every
    //      block's scaled-MC integer base + subpel ends up shifted.
    //      Fix: pass `ref_buf.width() / height()` and the raw header
    //      dims. Lights up the Y plane on all 7 mismatched frames.
    //
    //   2. The Q4 sub-pel sum bundled the plane-pixel coord and MV
    //      via one `scaled_x_q4` call for all planes. Libvpx splits
    //      these: `scaled_buffer_offset` uses the *plane* coord
    //      (chroma = luma>>1) for the integer base, but `vp9_scale_mv`
    //      uses the *luma* coord (`mi_x + x`) for the sub-pel mask.
    //      For 4:2:0 chroma the two coords differ by 2×, so the sub-
    //      pel-Q4 result rotates by a different amount mod 16 and the
    //      kernel picks a different filter row — visible as off-by-1
    //      U/V on every chroma block when scaling is active. Fix
    //      mirrors libvpx's split exactly inside the `is_scaled()`
    //      branch; unscaled refs keep the cheap unified path.
    //
    // Cascade: vp90-2-05-resize 3→10 (+7). Conformance 168/23/10 →
    // 175/16/10. Remaining 16 mismatches: tile_1x4 8 + tile_1x8 8 (a
    // multi-tile entropy-isolation bug specific to >2 tiles per row,
    // D.23's lane).
    // D.22 (2026-05-19): fix MV-component hp count gating + altref
    // prev_frame swap. Two distinct bugs, both surfaced via libvpx
    // INTERMI altref trace on `vp90-2-09-aq2.webm`:
    //
    //   1. `read_mv_component` was gating the `hp` count bump on
    //      `use_hp`. Libvpx's `inc_mv_component` (`vp9_entropymv.c:111-136`)
    //      hard-codes `usehp=1` at the call site (`vp9_inc_mv` at :145
    //      / :149). When a block's `use_hp=false`, the hp bit is
    //      hard-coded to 1 at decode time, but `comp_counts->hp[1]`
    //      *still gets bumped* — the count reflects the EFFECTIVE hp
    //      bit. Our gating under-counted `hp[1]`, drifting `comp.hp`
    //      and `comp.class0_hp` away from libvpx after every frame
    //      that adapts MV probs with `allow_high_precision_mv` ON.
    //
    //   2. `decode_inter_frame` was pulling `prev_mv_grid` from
    //      `mv_grids[ref_frame_indices[0]]` (the LAST_FRAME ref-map
    //      slot's grid). Libvpx's `cm->prev_frame->mvs` follows the
    //      `cm->prev_frame = cm->cur_frame` swap (`vp9_decoder.c:478`)
    //      that fires after EVERY non-show-existing decode —
    //      independent of which `ref_frame_map` slot was refreshed.
    //      For altref frames whose LAST_FRAME index points at a slot
    //      holding an older frame's grid, the two diverge. Added
    //      `Decoder::prev_frame_mv_grid` + `prev_frame_mv_grid_dims`
    //      to track the most-recently-decoded grid directly.
    //
    // Bisect: libvpx-side detailed `read_mv` + `read_mv_component`
    // tracing (RVP9_TRACE_RDMV, RVP9_TRACE_RDMV_COMP) on aq2 `f=16`
    // (the 2nd altref) pinpointed `comp[0].hp` diverging (libvpx 99,
    // rvp9 131) — same input range/value/count/probs from `joints`
    // down through `bits[i]`, only `hp` differed. The HP prob mismatch
    // shifted bits at `vpx_read(hp_prob)` causing range/count to
    // diverge in the col decode that follows — final MV `(-32,-106)`
    // vs `(-32,-98)` (col differed by 8 = 1 pel).
    //
    // Cascade scope: aq2 14→100 (+86), sharpness-1 3→10 (+7),
    // sharpness-7 3→10 (+7), tile_1x2 2→10 (+8), show-existing-frame
    // 1→5 (+4). Conformance 56/135/10 → 168/23/10 (+112 frames).
    // Remaining 23 mismatches: resize 7, tile_1x4 8, tile_1x8 8.
    // D.21 (2026-05-19): fix `size_group` mapping in
    // `read_intra_in_inter_mode_info` — Block8x8 was wrongly mapped to
    // size_group=0, Block16x16/Block32x32 cohorts were each one index
    // too low. Libvpx `size_group_lookup` (vp9_common_data.c:32-33)
    // is `{0,0,0, 1,1,1, 2,2,2, 3,3,3,3}` indexed by BLOCK_SIZE; we
    // had `{0,0,0,0, 1,1,1, 2,2,2, 3,3,3}`. The default-arm intra Y
    // mode read for non-keyframe intra blocks uses
    // `fc.y_mode_prob[size_group]`; the wrong row produced a different
    // y_mode and desynced the bool-coder cursor on aq2 segmentation
    // streams (and any inter stream with an intra block ≥ Block8x8).
    // Confirmed via libvpx altref INTERMI trace at aq2 mi=(5,0):
    // libvpx post_modeinfo range=216 count=29 (mode=DC); rvp9 read
    // range=232 count=28 (mode=H). Same cursor entry, different bits
    // consumed ⇒ wrong prob row. The D.20 altref seg_id mismatch at
    // mi=(0,8) was the visible end of this cascade — the cursor
    // started drifting at mi=(5,0)'s intra Y mode read. After the
    // fix: aq2 lights up plus 24 other inter vectors that exercise
    // intra-in-inter blocks (aq2 segmentation, altref-driven inter,
    // most quantizer / sharpness / deltaq / resize sweeps).
    // Conformance 31/160/10 → 56/135/10. Required libvpx-stock
    // build fix (TMPDIR=C:/tmp, examples-x86_64-win64-gcc.mk + libs-
    // *.mk + gxx_wrap.sh) and drop of cm->show_frame filter on
    // RVP9_TRACE_INTERMI to capture the altref trace.
    // D.20 (2026-05-19): NEGATIVE — diagnostic infra only. Ratchet
    // unchanged. Localized aq2 frame-2 (64, 0) cascade to altref's
    // mi=(0,8) seg_id divergence (rvp9 decodes seg=1; libvpx altref
    // produces seg=0 per the frame-2 inherit-via-min trace at
    // libvpx_imi.txt). Confirmed bool-coder cursor diverges inside
    // altref's SB(0,8) syntax — cannot pin to a specific bit without
    // a libvpx altref INTERMI trace (build env wedged). Three new
    // env hooks for D.21: `RVP9_DUMP_ALTREF=<path>` for altref Y
    // plane dump, `RVP9_TRACE_SEG_ID=1` for per-block seg-id reads,
    // `RVP9_TRACE_SEG=1` extended to dump first-SB-row seg map.
    // D.15 (2026-05-18): sub-8 luma `dec_reset_skip_context` now pins
    // the entropy-context span to the parent 8×8 mi cell (n4 = 2 in
    // each axis), matching libvpx's `set_plane_n4(bwl=bhl=1)` for any
    // sub-8 child of a Block8×8 parent. Previously we used
    // `plane_bsize.n_4x4_*` which is 1×1 / 1×2 / 2×1 / 2×2 for
    // sub-8 — leaving one cell of `left_coef` un-cleared on every
    // sub-8 skip, which poisoned `initial_pt` for the next
    // decode_coefs at the same SB-row y-offset. Lights up all 6 of the
    // failing 352×288/352×240 keyframes (q26, q63, sharpness-1/7,
    // deltaq, resize) — frame 1 of each now MD5-matches libvpx. The
    // 2-tile keyframes (1x2 / 1x4 / 1x8) already matched (no sub-8
    // luma blocks in those frames). Conformance 22/169/10 → 31/160/10.
    // E.2 (2026-05-18): single-threaded multi-tile decode wired up.
    // Removes the `MultiTileUnsupported` reject on the 1x2/1x4/1x8
    // tile vectors (30 previously-skipped frames). All 30 now DECODE
    // (no panic, no error) but MD5-mismatch — expected, since these
    // vectors exercise tile-boundary entropy isolation that the floor
    // is now exposed to for the first time. `MAX_MISMATCHES` rises
    // from 139 → 169 (the new mismatches) and `total_skipped` falls
    // from 40 → 10. `MIN_PASSES` unchanged. Also fixes a pre-existing
    // `1u16 << 16` overflow in `adjust_mask` for `rows == 7`
    // (triggered by the 1920×1080 tile vectors).
    // D.9 (2026-05-18): backward-adapt now uses a pre-compressed-header
    // FrameContext snapshot as the `pre` baseline (mirroring libvpx's
    // `pre_fc = cm->frame_contexts[fc_idx]` while compressed-header
    // diff-updates land on `cm->fc` — a working copy our code didn't
    // model). Without the snapshot, every inter frame's adapt was
    // chaining from `bank + this_frame's_diff_updates`, accumulating
    // probability-table drift that corrupted the bool-coder cursor
    // after several frames. Conformance moves from 11/150/40 →
    // 22/139/40 — `vp90-2-02-size-08x08.webm` lights up frames 5-10
    // (now full 10/10) and `vp90-2-02-size-32x32.webm` lights up
    // frames 5-9 (9/10, only frame 10 still mismatches).
    // D.8 (2026-05-18): wire the loop-filter apply driver into the
    // inter-frame path (matches the K2.c keyframe wireup). libvpx's
    // `vpxdec --i420` reference output is post-LF; without this the
    // inter frames were being MD5'd as pre-LF. Conformance moves from
    // 8/153/40 → 11/150/40 — three additional inter frames light up,
    // including frames 2/3/4 of `vp90-2-02-size-32x32.webm`.
    // D.7 (2026-05-18): inter-frame cascade work (mv_ref + decoder
    // ref-pool / tile decode tweaks) lights up 2 additional frames.
    // Conformance moves from 6/155/40 → 8/153/40.
    // M.15 (2026-05-18): added libvpx 4x4 specializations for
    // D45 / D63 / D207 intra-pred. Conformance: 3/158/40 → 6/155/40.
    // D.5 (2026-05-18): `extend_mc_border` scratch-buffer fallback.
    // Conformance: 2/159/40 → 3/158/40.
    // D.25 (2026-05-19): Stage 3 EXIT. Fixed partial-bottom / partial-
    // right residual-decode loop clamp in `reconstruct_inter_block` —
    // libvpx's `predict_recon_inter` (`vp9_decodeframe.c:892-921`) gates
    // the per-4×4-block iteration on `max_blocks_{wide,high}` derived
    // from `mb_to_{right,bottom}_edge`; rvp9 was iterating the full
    // plane extent and reading coef tokens for the invisible-edge TX
    // cells, drifting the bool-coder cursor relative to libvpx for any
    // partial-row/col inter block. Surfaced on tile_1x4 / tile_1x8 at
    // 1920×1080 / 3840×2160 where the partial-bottom SB row + altref
    // compound reference let the cursor drift cascade visible into
    // frame 3 / 5 / 7 / etc. via MC. Lights up the final 16 mismatches
    // (8 tile_1x4 + 8 tile_1x8) → 175/16/10 → 191/0/10.
    // Pre-D.3.a baseline: 1/9/191.
    assert!(
        total_passed >= MIN_PASSES,
        "regression: expected at least {MIN_PASSES} passed, got {total_passed}",
    );
    assert_eq!(
        total_mismatched, 0,
        "stage 3 exit: any mismatch is a regression",
    );
    // K-5 (HLD V6 §G1, lines 1806-1821): hard-fail any non-
    // `Unimplemented` decode error. Pre-K-5 the harness folded these
    // silently into `total_skipped`, masking real regressions behind
    // the expected-skip count.
    assert_eq!(
        total_other_err, 0,
        "conformance harness saw {total_other_err} non-Unimplemented \
         decode errors; pre-K-5 the silent skip-folding bug would \
         have masked these as expected skips",
    );
}

/// Unit tests for the inline-reparse `expected_sub_frame_count` oracle.
///
/// G8.0 review (2026-05-21): pins the inline implementation against
/// hand-crafted trailing-marker bytes so a future refactor that
/// accidentally re-delegates to `parse_superframe_index` (and thus re-
/// introduces the tautology) is caught at unit-test time.
#[cfg(test)]
mod expected_sub_frame_count_tests {
    use super::expected_sub_frame_count;

    /// Empty packet → fallback count of 1 (matches the comment in
    /// `expected_sub_frame_count`).
    #[test]
    fn empty_packet_is_one() {
        assert_eq!(expected_sub_frame_count(&[]), 1);
    }

    /// A non-marker trailing byte → single-frame packet, count = 1.
    #[test]
    fn non_marker_trailer_is_one() {
        // 0x00 — top 3 bits are 000, not 110.
        assert_eq!(expected_sub_frame_count(&[0x12, 0x34, 0x00]), 1);
        // 0xff — top 3 bits are 111, not 110.
        assert_eq!(expected_sub_frame_count(&[0x12, 0xff]), 1);
        // 0xa0 — top 3 bits are 101.
        assert_eq!(expected_sub_frame_count(&[0xa0]), 1);
    }

    /// Marker bytes covering every 1..=8 sub-frame count, with mag bits
    /// varied. The mag bits do NOT affect the count derivation, so the
    /// oracle must return the right count regardless.
    #[test]
    fn marker_low_three_bits_decode_to_count_plus_one() {
        // (count, marker) pairs.
        for count in 1usize..=8 {
            // mask = low three bits of marker → count - 1 (0..=7).
            let mask = (count - 1) as u8;
            // Try mag = 1..=4 (mag bits at marker[3..=4]).
            for mag_bits in 0u8..=3 {
                let marker = 0xc0 | (mag_bits << 3) | mask;
                // Single trailing byte is enough — the oracle only
                // reads `packet.last()`. A real superframe would have
                // a longer body, but the oracle is intentionally
                // agnostic to that.
                let packet = [marker];
                assert_eq!(
                    expected_sub_frame_count(&packet),
                    count,
                    "marker=0x{marker:02x} (count={count} mag_bits={mag_bits})",
                );
            }
        }
    }

    /// The boundary cases: 0xc0 (count=1, mag=1) and 0xdf (count=8,
    /// mag=4) — the lowest and highest legal marker bytes.
    #[test]
    fn marker_boundaries() {
        assert_eq!(expected_sub_frame_count(&[0xc0]), 1);
        assert_eq!(expected_sub_frame_count(&[0xdf]), 8);
    }

    /// 0xe0 sits just outside the marker range (top 3 bits = 111) — it
    /// must NOT be misclassified as a superframe.
    #[test]
    fn just_above_marker_range_is_single() {
        assert_eq!(expected_sub_frame_count(&[0xe0]), 1);
        assert_eq!(expected_sub_frame_count(&[0xbf]), 1);
    }
}
