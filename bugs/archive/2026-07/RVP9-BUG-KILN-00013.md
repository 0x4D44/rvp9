# RVP9-BUG-KILN-00013 - ALTREF frames disable prev-MV salvage while the decoder enables it

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** encoder/altref prev-MV salvage
- **Raised:** 2026-07-05
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T170438Z-p91782-n976304000-c1 branch=task/bug-RVP9-BUG-KILN-00013-run-fix-20260730T170438Z-p91782-n976304000-c1 code=d39554d2af295361751556dd614afc786b37dfa0 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`crates/rvp9-encoder/src/frame_loop.rs:3400` passes `None` as the ALTREF picker's previous-MV grid, and `crates/rvp9-encoder/src/frame_loop.rs:3511` passes `None` to the ARF packet writer. The decoder computes `use_prev_frame_mvs` structurally at `crates/rvp9-decoder/src/lib.rs:2931`, using the previous decoded frame's show/key state. For an ARF decoded after a shown inter frame, that gate can be open.

The ARF path can code NEWMV candidates; the inter mode order includes NEWMV cases in `crates/rvp9-core/src/mode_decision/inter.rs:1431` and `crates/rvp9-core/src/mode_decision/inter.rs:1434`. If the encoder writer derives the NEWMV residual against no previous grid while the decoder derives the best reference MV with the previous grid, the decoded reconstruction can drift.

## Fix

Open. The ARF picker and writer should mirror the decoder's structural previous-MV predicate for the ARF being encoded, while still closing the following show frame's previous-MV gate after the non-show ARF. Add a non-ignored round-trip or cross-decoder regression that forces an ALTREF-containing stream with NEWMV-capable content.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Root cause addressed: both the `None` at the ARF picker and the `None` at the ARF packet
writer are replaced by the same `arf_prev_mv_grid` (`crates/rvp9-encoder/src/frame_loop.rs:3393`,
`:3440`, `:3549`). The encoder's new predicate was compared term-by-term against the decoder's
at `crates/rvp9-decoder/src/lib.rs:3028`; the two omitted terms are provably safe -- the ARF
packet is always written `error_resilient = false`, and the encoder emits no VP9 intra-only
frames.
The regression test is a real oracle, not a witness check:
`altref_reconstruction_matches_decoder_with_prev_mv_salvage` compares the encoder's ARF
reconstruction byte-for-byte against the DECODER's, and separately asserts the stream actually
codes NEWMV and that `diag_last_arf_used_prev_mv_grid() == Some(true)`, so it cannot pass
vacuously. Observed passing.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
