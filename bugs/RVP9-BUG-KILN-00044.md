# RVP9-BUG-KILN-00044 - WebM reader drops laced blocks and ignores track identity

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** io/webm packet reader
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T173030Z-p2642-n978179000-c1 branch=task/bug-RVP9-BUG-KILN-00044-run-fix-20260730T173030Z-p2642-n978179000-c1 code=8bd3eb2c669e76e421c26080dfb75b8f61634c9f gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The WebM reader does not reliably return the VP9 video packet stream it advertises.

`crates/rvp9-io/src/container.rs:315`-`317` says the parser does not validate codec ID, track number, or metadata, while `read_webm_packets` documents returned ranges as video-track blocks at lines `380`-`381`. The walker accepts every `SimpleBlock` or `Block` inside a `Cluster` at lines `442`-`453`. `parse_block_frame` reads the track number into `_track` at line `498` and discards it, so audio, subtitle, or non-VP9 blocks can be handed to the VP9 decoder as packets.

Unsupported lacing goes the other direction and silently drops data. Lines `510`-`514` return `Ok(None)` for any non-zero lacing mode, and `crates/rvp9-io/tests/error_paths.rs:648`-`665` pins that as a no-error/no-packet outcome. `rvp9dec` reads the packet list at `crates/rvp9-cli/src/bin/rvp9dec.rs:399` and feeds each returned range into `parse_packet` at lines `432`-`436`, so both the CLI and harness users inherit the wrong packet set.

## Fix

Open. Parse `Tracks`/`CodecID` enough to identify the VP9 video track and filter to it, or document and enforce a single-track VP9-only subset. Reject laced blocks as unsupported unless lacing is implemented.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. This is adjacent to harness skip-accounting bugs, but not a duplicate: the parser itself can return the wrong packet stream.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Both halves are fixed at the root: `Tracks`/`TrackEntry` is now parsed for TrackNumber +
`TrackType == 1` + `CodecID == "V_VP9"`, the discarded `_track` became a real filter, and lacing
on the selected track is an error instead of a silent `Ok(None)` drop
(`crates/rvp9-io/src/container.rs:962`, `:1186-1188`, `:1189-1193`), mirrored in the newer
streaming reader at `:666` and `:854`.
`webm_non_vp9_track_is_rejected`, `webm_filters_other_tracks_and_returns_only_the_vp9_track`,
`webm_bare_block_with_lacing_inside_cluster_is_rejected`, `webm_block_with_lacing_set_is_rejected`
and `container::tests::webm_parse_block_lacing_is_rejected` all observed passing.
Over-strict-guard check -- this was the highest-risk item in its batch and could NOT be settled
from a bare worktree, because a shipped vector lacking `V_VP9`/`TrackType=1` would turn into a
mass conformance failure. Settled empirically for this verification by fetching the official
vectors: with 12 real `.webm` conformance vectors present, all 14 `vec_*` tests and
`conformance_runner_walks_all_vectors` ran and passed. The new track filter accepts the real
corpus.
Residual noted, not blocking: a WebM with no `Tracks` element at all now hard-errors instead of
yielding zero packets -- a behaviour change beyond the report, with no
`notes/divergence-ledger.md` entry; recorded in `scratchpad.md`.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
