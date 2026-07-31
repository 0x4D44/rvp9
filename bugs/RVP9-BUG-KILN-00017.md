# RVP9-BUG-KILN-00017 - Public segmentation helper panics on out-of-range segment IDs

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/segmentation public API
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T194323Z-p38606-n589160000-c1 branch=task/bug-RVP9-BUG-KILN-00017-run-fix-20260730T194323Z-p38606-n589160000-c1 code=4ae37be1da2947c526eddfc1323ef534d55341f5 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`crates/rvp9-encoder/src/lib.rs:121` exports `pub mod segmentation`. The public `choose_segmap_coding_method` function accepts caller-provided segment maps at `crates/rvp9-encoder/src/segmentation.rs:188`. It converts `cur_segmap[i]` to `cur_id`, only `debug_assert`s `cur_id < MAX_SEGMENTS` at `crates/rvp9-encoder/src/segmentation.rs:218`, then indexes `no_pred_segcounts[cur_id]` at `crates/rvp9-encoder/src/segmentation.rs:219`.

Release callers can pass segment IDs `>= 8` and panic the process.

## Fix

Open. Validate every segment ID in normal code before indexing. Either return a `Result<SegmapMethod, ...>` or reject invalid maps before tallying.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Correct layer: validation runs before any tally can index the fixed-size `[u32; MAX_SEGMENTS]`
arrays (`crates/rvp9-encoder/src/segmentation.rs:240-256`), and the `debug_assert` that was the
only guard is removed rather than left as a redundant release no-op.
`segmentation::tests::choose_segmap_method_rejects_out_of_range_ids` observed passing. It could
not compile against the pre-fix signature (which returned a bare `SegmapMethod` with no error
type), so it cannot be a false green.
Residuals noted, not blocking: the function still panics on a length mismatch between the two
maps, and `choose_segmap_coding_method` has no in-crate caller -- this is hardening on an API
nothing currently uses.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
