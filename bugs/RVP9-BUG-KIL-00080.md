# RVP9-BUG-KIL-00080 — Parallel decoder retains one full-frame scratch buffer per tile

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** core/frame decoder tile scratch
- **Raised:** 2026-08-22T09:00:46Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-22T09:00:46Z, raised via `deltic bugs new`)

## Observation

Static source review found that parallel decode sizes its scratch allocation by tile count rather than active worker count. The gate at `crates/rvp9-decoder/src/lib.rs:1803-1826` permits one tile row with multiple columns and more than one thread. It then sets `num_tiles = buffers.len()` at `:1887` and calls `Yv12ScratchPool::take(num_tiles, &yv12)` at `:1924-1927`. Each scratch is a full-geometry `Yv12Buffer`; the core pool documents about 3 MiB per 1080p scratch at `crates/rvp9-core/src/frame.rs:967-980`, and it retains the maximum count seen for the decoder lifetime at `:986-993` and `:1103-1117`.

The tile parser caps a grid at 64 columns by 4 rows, and this parallel path requires one row, so a valid frame can still select 64 tile columns. With two decode threads, only two jobs can execute concurrently but the decoder allocates and retains 64 full-frame buffers. That is about 192 MiB of scratch at 1080p and scales with supported frame geometry. If allocation fails, the `.expect("scratch Yv12Buffer allocation")` at `lib.rs:1927` turns pressure into a process panic.

Expected: peak and retained full-frame scratch scale with the bounded number of active workers. Actual: it scales with attacker-controlled valid tile count, up to 64 buffers on the reachable parallel path.

## Proposed resolution

Allocate at most `min(num_threads, num_tiles)` reusable full-frame scratch buffers and schedule tiles through those worker-local buffers, or allocate tile-local geometry instead of full frames. Propagate allocation failure as `DecodeError` rather than panicking. Add an inspectable allocation-count regression for 64 tile columns with two threads and a retained-capacity assertion. Estimated effort: medium.

## Review evidence

Confirmed independently by the security lens, the lead's source trace, and an adversarial reliability recheck. This is distinct from open RVP9-BUG-KILN-00070, which tracks the encoder's separate parallel-tile scratch allocation. No decoder or allocation test was run in this static review pass.

## Fix

<unfixed — raised only>

## Notes
