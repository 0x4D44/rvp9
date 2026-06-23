# RVP9-BUG-KILN-00008 - G-C decode_inter_block integration paths (SEG_LVL_REF_FRAME force glue + Sub8SegSkip Err path) lack end-to-end coverage

- **State:** Closed
- **Severity:** Low
- **Area:** decoder/segmentation test-coverage
- **Raised:** 2026-06-19 by Claude overnight code review
- **State history:** 2026-06-19 Opened by Claude from the track2 decoder-fix harvest review pass. 2026-06-21 Fixed by Claude (task/fix-seg-coverage): added `seg_skip_sub8_through_decode_inter_block_errs_without_advancing` in `crates/rvp9-decoder/src/tile.rs`, driving the real `decode_inter_block` end-to-end for (b) the Sub8SegSkip Err path (asserts `Err(Sub8SegSkip)` + bool reader does not advance) and (a) the SEG_LVL_REF_FRAME `is_inter` derivation (both branches: `Some(LAST)`→inter→guard fires; `Some(INTRA)`→`is_inter=false`→intra path, guard correctly does not fire). Verified red-before-green by weakening the guard predicate. Test-only; no version bump. Awaiting two-eyes verification → Closed.
- **State history:** 2026-06-23 Closed by Codex (task/20260623-TSK-HUM-verify-fixed-bugs): independently verified the original coverage gap is closed by the real `decode_inter_block` integration test. `$null | mdtimeout 180 cargo test -p rvp9-decoder --lib seg_skip_sub8_through_decode_inter_block_errs_without_advancing -- --nocapture` passed, covering the Sub8SegSkip Err path, no-reader-advance invariant, and forced LAST/INTRA `SEG_LVL_REF_FRAME` branch derivation.

## Resolution note (2026-06-21)

**Covered end-to-end through `decode_inter_block`:** the Sub8SegSkip guard Err path
(the new G-C fallible flow) + the `seg_ref` lookup and `is_inter` derivation (both
branches). Reachable cleanly because, with `seg_ref=Some` + `seg_skip` active +
`segmentation_update_map=false`, segment_id is inherited, skip is forced, and
`is_inter` is derived — all with zero bits read — so the guard is hit at the reader's
start position and the ref pool is never accessed (`refs:[None;3]`).

**Residual (not covered; disproportionate for Low):** the post-guard forced `ref0`
derivation (`seg_ref.and_then(from_i8)`, tile.rs:2503) and full inter-on-REF_FRAME
reconstruction run past the guard into motion compensation, needing real reference
frames (a motion-comp harness or a crafted fixture). That path stays covered by the
rvp9-core unit oracle (`read_ref_frames_seg_force_consumes_zero_bits`). The in-code
`TODO(G-C verify)` (tile.rs:2359-2366) asks specifically for a *differential* fixture
placing inter blocks on a REF_FRAME segment — still open (libvpx's realtime nonrd
encoder never produces such content, so it needs a crafted stream). A two-eyes
verifier may keep this open if differential coverage of the ref0 path is required.

## Observation

Two decode-path branches introduced by G-C (`b245c53`) have no test that drives
them through `decode_inter_block`:

**(a) SEG_LVL_REF_FRAME glue.** The segment-id index
`params.base.seg_ref_frame[segment_id as usize]`
(`crates/rvp9-decoder/src/tile.rs:2368`), the is_inter derivation
`ref_idx != RefFrame::IntraFrame as u8` (`crates/rvp9-decoder/src/tile.rs:2370`),
and `seg_ref.and_then(|idx| RefFrame::from_i8(idx as i8))` threaded into
`block_inter_params` (`crates/rvp9-decoder/src/tile.rs:2503`). The differential ROI
fixture cannot reach it — the author's own `TODO(G-C verify)` at
`crates/rvp9-decoder/src/tile.rs:2359-2366` and the fixture doc at
`crates/rvp9-harness/tests/segmentation_segfeatures.rs:23-29` note that libvpx's
realtime nonrd encoder never lands an inter block on the `REF_FRAME` segment. The
`rvp9-core` unit oracles (`read_ref_frames_seg_force_consumes_zero_bits`,
`seg_skip_forces_zeromv_and_consumes_zero_bits`) call `read_ref_frames` /
`read_inter_block_mode_info` **directly** with hand-built `InterModeReadParams`,
bypassing this glue. The forced-INTRA case (`seg_ref == Some(0)` → `is_inter = false`
→ intra-in-inter path) is wholly un-exercised.

**(b) Sub8SegSkip guard Err path.** The guard
`if seg_skip_active && (bsize as u8) < (BlockSize::Block8x8 as u8) { return
Err(TileDecodeError::Sub8SegSkip{..}) }` (`crates/rvp9-decoder/src/tile.rs:2496`)
and the `?`-propagation Err path it relies on. The only test,
`sub8_seg_skip_is_a_decode_error`, hand-constructs the `Sub8SegSkip` variant and
asserts its `Display` text plus that `Block4x4/4x8/8x4` sort below `Block8x8`; it
never calls `decode_inter_block`, so it cannot catch a guard that fails to fire or
fires on the wrong predicate. (The **Ok** path of the new `?` chain *is* exercised
by `roi_segfeatures_decodes_bit_exact_vs_vpxdec` on legal ≥8×8 blocks — only the
Err trigger is uncovered.)

Both branches are **correct as written** (verified libvpx-faithful:
`RefFrame::IntraFrame == 0`; `RefFrame::from_i8` is total over the parser-clamped
`0..=3` range so the `.and_then` never returns `None`; the guard mirrors
`vp9_decodemv.c:709-712` and returns before any reader advance). This is a
test-coverage gap, not a behavioural defect.

## Notes

Two findings (F3 SEG_LVL_REF_FRAME glue, F4 Sub8SegSkip Err path) folded into one
entry — same root cause (the unit oracles bypass `decode_inter_block`; the
differential fixture never lands on these branches) and the same fix shape. 0/3
skeptics refuted either; both downgraded to **low** (correct, libvpx-faithful,
effectively dead on all current real-world/conformance content, author-documented
TODO). One secondary worry was checked and dismissed: the early `return Err`
skipping `part_ctx.update` is harmless because the error is a terminal whole-frame
abort and `part_ctx` is per-tile scratch discarded on abort.

**Fix:** either mint a fixture that places inter blocks (and an intra-forced block
via `seg_ref = INTRA_FRAME`) on a `SEG_LVL_REF_FRAME` segment, or add a
`decode_inter_block`-level integration test that sets
`params.base.seg_ref_frame[seg]` / `seg_skip[seg]` and asserts: the derived
`is_inter`, the forced `ref0`, the `Sub8SegSkip` `Err` on a sub-8 block, and that
the bool reader did not advance past the guard. Resolves the in-code
`TODO(G-C verify)` at `tile.rs:2359-2366`.
