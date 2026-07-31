# RVP9-BUG-KILN-00028 - parse_packet commits stream metadata before frame decode succeeds

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/packet parse state
- **Raised:** 2026-07-07
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T175756Z-p1788-n335904000-c1 branch=task/bug-RVP9-BUG-KILN-00028-run-fix-20260730T175756Z-p1788-n335904000-c1 code=35238a7a6bec9655f766f3d3a74818c03037a5d4 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`Decoder::parse_packet` mutates stream history while it is only parsing headers. It overlays and updates `established_chroma` at `crates/rvp9-decoder/src/lib.rs:1063` and calls `refresh_ref_frame_sizes` at `crates/rvp9-decoder/src/lib.rs:1080`. That refresh updates the size cache for future `setup_frame_size_with_refs` header parsing.

The actual decoded reference-map commits happen later, only after frame decode succeeds: intra/keyframe refresh writes ref slots at `crates/rvp9-decoder/src/lib.rs:2420`, and inter refresh writes ref slots at `crates/rvp9-decoder/src/lib.rs:3876`.

A caller can parse a packet and drop it, or parse frames whose later decode fails, and still advance future header parsing to dimensions or chroma from frames that never landed in the reference map. Superframes make this sharper because all subframes are parsed before any decode commit.

## Fix

Open. Parse packet headers through local shadow copies of `ref_frame_sizes` and `established_chroma`. Commit those deltas only alongside the successful frame reference-map commit, or parse/decode superframes sequentially so parse state cannot outrun decode state.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. Track Q already flags parse-time mutable state for future frame-parallel work, but this malformed/error-path state-commit bug was not in the bug ledger.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Exactly the shadow-then-commit shape this record asked for:
`refresh_ref_frame_sizes` became a free function over a caller-supplied array, so parse writes
only a local copy and the commit happens after decode succeeds
(`crates/rvp9-decoder/src/lib.rs:1083-1122`, `:1193-1202`, called at `:2586` and `:4058`).
Confirmed `decode_intra_frame_inner` and `decode_inter_frame` each have exactly ONE `Ok(...)`
exit with the commit immediately above it, so no success path skips it, and
`decode_show_existing_frame` correctly does not commit.
`tests::parse_and_failed_decode_do_not_commit_stream_metadata` (the real regression),
`::commit_stream_metadata_keyframe_sets_every_size_slot` and
`::commit_stream_metadata_intra_only_honours_size_mask` all observed passing. The first fails
pre-fix, because `parse_packet` used to write both caches immediately.
Behaviour-change check: a caller that parses packet N+1 before decoding packet N would now see
stale sizes. All six in-tree callers were traced and every one is strict
parse-then-decode-per-packet; none regresses.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
