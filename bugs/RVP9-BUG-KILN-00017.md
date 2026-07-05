# RVP9-BUG-KILN-00017 - Public segmentation helper panics on out-of-range segment IDs

- **State:** Open
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`)

## Observation

`crates/rvp9-encoder/src/lib.rs:121` exports `pub mod segmentation`. The public `choose_segmap_coding_method` function accepts caller-provided segment maps at `crates/rvp9-encoder/src/segmentation.rs:188`. It converts `cur_segmap[i]` to `cur_id`, only `debug_assert`s `cur_id < MAX_SEGMENTS` at `crates/rvp9-encoder/src/segmentation.rs:218`, then indexes `no_pred_segcounts[cur_id]` at `crates/rvp9-encoder/src/segmentation.rs:219`.

Release callers can pass segment IDs `>= 8` and panic the process.

## Fix

Open. Validate every segment ID in normal code before indexing. Either return a `Result<SegmapMethod, ...>` or reject invalid maps before tallying.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.
