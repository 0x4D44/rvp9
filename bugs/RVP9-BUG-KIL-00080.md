# RVP9-BUG-KIL-00080 — Parallel decoder retains one full-frame scratch buffer per tile

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** core/frame decoder tile scratch
- **Raised:** 2026-08-22T09:00:46Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T070603Z-845b2d74
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00080-run-verify-20261019T070603Z-845b2d74
- **Owner base:** bd6aca5027933a306bae50474ef3692e978b3106
- **Owner fingerprint:** sha256:0528b64aad69824db233596811cd258c03a4c22117ae40ccb0ff39f59db365c4
- **Owner since:** 2026-10-19T07:06:03Z
- **Owner until:** 2026-10-19T11:06:03Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T09:00:46Z, raised via `deltic bugs new`) -> Fixed (2026-08-22T13:08:16Z, deltic:auto role=fix run=fix-20260822T130204Z-1abda903 branch=task/bug-RVP9-BUG-KIL-00080-run-fix-20260822T130204Z-1abda903 code=706f1637 gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260822T130204Z-1abda903 branch=task/bug-RVP9-BUG-KIL-00080-run-fix-20260822T130204Z-1abda903 code=dc19bcea3f41952f6d0a2a53b027c852a79a5aed gate=manual provenance-only=true reconciled-by=Codex)

## Observation

Static source review found that parallel decode sizes its scratch allocation by tile count rather than active worker count. The gate at `crates/rvp9-decoder/src/lib.rs:1803-1826` permits one tile row with multiple columns and more than one thread. It then sets `num_tiles = buffers.len()` at `:1887` and calls `Yv12ScratchPool::take(num_tiles, &yv12)` at `:1924-1927`. Each scratch is a full-geometry `Yv12Buffer`; the core pool documents about 3 MiB per 1080p scratch at `crates/rvp9-core/src/frame.rs:967-980`, and it retains the maximum count seen for the decoder lifetime at `:986-993` and `:1103-1117`.

The tile parser caps a grid at 64 columns by 4 rows, and this parallel path requires one row, so a valid frame can still select 64 tile columns. With two decode threads, only two jobs can execute concurrently but the decoder allocates and retains 64 full-frame buffers. That is about 192 MiB of scratch at 1080p and scales with supported frame geometry. If allocation fails, the `.expect("scratch Yv12Buffer allocation")` at `lib.rs:1927` turns pressure into a process panic.

Expected: peak and retained full-frame scratch scale with the bounded number of active workers. Actual: it scales with attacker-controlled valid tile count, up to 64 buffers on the reachable parallel path.

## Proposed resolution

Allocate at most `min(num_threads, num_tiles)` reusable full-frame scratch buffers and schedule tiles through those worker-local buffers, or allocate tile-local geometry instead of full frames. Propagate allocation failure as `DecodeError` rather than panicking. Add an inspectable allocation-count regression for 64 tile columns with two threads and a retained-capacity assertion. Estimated effort: medium.

## Review evidence

Confirmed independently by the security lens, the lead's source trace, and an adversarial reliability recheck. This is distinct from open RVP9-BUG-KILN-00070, which tracks the encoder's separate parallel-tile scratch allocation. No decoder or allocation test was run in this static review pass.

## Fix

Historical fix provenance is reconciled below. Independent behavioral verification remains pending.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `dc19bcea3f41952f6d0a2a53b027c852a79a5aed` (fix(dec): bound tile scratch by worker count).
The original reference `706f1637` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `4640da6d67a4c6b027ae275e01403079a805e5e9` has this fixing commit as its immediate parent.
Replaces both parallel tile gates with parallel_tile_decode_enabled requiring num_tiles <= num_threads; converts both scratch-pool allocation expects to DecodeError::FrameAlloc propagation; adds parallel_tile_scratch_is_bounded_by_worker_count with 64 tiles/two workers case.
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

## Notes
