# RVP9-BUG-KILN-00023 - Public tile buffer helpers trust invalid tile counts

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/bitstream tile helpers
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`)

## Observation

`crates/rvp9-core/src/bitstream/tile.rs:24` documents the VP9 tile limit, but public `parse_tile_buffers` enforces only zero-dimension rejection before computing `tile_cols * tile_rows` and allocating `Vec::with_capacity(total)` at `crates/rvp9-core/src/bitstream/tile.rs:70`. Its docs claim no panic at `crates/rvp9-core/src/bitstream/tile.rs:58`, but hostile public callers can still request unbounded helper allocation.

The paired `write_tile_buffers` helper computes the expected payload count at `crates/rvp9-core/src/bitstream/tile.rs:145`, but checks it only with `debug_assert_eq!` at `crates/rvp9-core/src/bitstream/tile.rs:146`. Release builds emit data based on the actual payload list at `crates/rvp9-core/src/bitstream/tile.rs:156`, even when that count does not match the supplied tile geometry.

## Fix

Open. Enforce VP9 tile bounds and payload counts in release code at the public helper boundary. Return typed errors for invalid tile dimensions or mismatched payload counts, and add tests for huge tile dimensions plus mismatched write counts.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review. Normal decoder header parsing appears to clamp tile counts before this helper, so the confirmed product impact is the public helper contract rather than the current decoder malformed-stream path.
