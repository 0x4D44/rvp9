# RVP9-BUG-KILN-00070 — Parallel tile scratch retains one full frame per tile

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** encoder/parallel scratch
- **Raised:** 2026-08-16T08:19:07Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261015T224158Z-d8168bcd
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00070-run-fix-20261015T224158Z-d8168bcd
- **Owner base:** 466fe287d36d5cc5d9af0ac3ca3f34d33cd7737c
- **Owner fingerprint:** -
- **Owner since:** 2026-10-15T22:41:58Z
- **Owner until:** 2026-10-16T02:41:58Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-16T08:19:07Z, raised via `deltic bugs new`)

## Observation

EncoderParallelism permits up to 64 tile columns and 64 threads. When a worker pool exists, D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\frame_loop.rs:2191 allocates full-frame YUV reconstruction, motion-vector, and partition grids for every tile, independent of active worker count. Later resizing at frame_loop.rs:4466 shortens the vectors but retains their full-frame capacities. At the supported 16,384 by 4,096 luma envelope, 64 YUV scratch sets alone retain about 6 GiB before the MV grids, partition grids, references, or main scratch.

## Fix

Allocate reconstruction and grid scratch for tile-local geometry or for the bounded number of active workers, not one full frame per tile. Make the allocation plan inspectable and add a non-OOM regression that proves retained capacity stays within the intended budget at the maximum supported geometry and tile count.

## Notes
