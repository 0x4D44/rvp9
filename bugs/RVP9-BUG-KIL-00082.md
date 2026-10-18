# RVP9-BUG-KIL-00082 — Parallel decoder retains full-frame metadata slabs per tile

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/parallel tile metadata
- **Raised:** 2026-08-22T10:24:29Z
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
- **State history:** Open (2026-08-22T10:24:29Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T02:34:07Z, deltic:auto role=fix run=fix-20261016T022804Z-9d7700c8 branch=task/bug-RVP9-BUG-KIL-00082-run-fix-20261016T022804Z-9d7700c8 code=dc19bcea3f41952f6d0a2a53b027c852a79a5aed gate=manual) -> Closed (2026-10-18T17:30:18Z, deltic:manual role=verify run=verify-20261018T172611Z-bacc71dc verified_fix_run=fix-20261016T022804Z-9d7700c8 verdict=close)

## Observation

The parallel intra path creates one `Vec<u8>` per tile at `crates/rvp9-decoder/src/lib.rs:1947-1963`, and every vector is sized to `mi_rows * mi_cols`. The merge reads only the tile rectangle at `:2280-2295`. The parallel inter path repeats that full-frame segment map and also creates one full-frame `Vec<PrevFrameMvRef>` per tile at `:3300-3331`; its merge likewise copies only the tile rectangle at `:3716-3732`.

Both pools persist on `Decoder` at `:555-570`, so the maximum tile-count allocation stays retained after the frame. Valid multi-column input therefore makes metadata memory and cold allocation scale as `O(tile_count * frame_area)`, although only the bounded active workers run concurrently and each tile accesses only its own rectangle.

Expected: worker metadata should scale with tile-local area or the number of active workers. Actual: the decoder retains one frame-wide segment slab per tile and, on inter frames, one frame-wide MV slab per tile.

## Fix

Store tile-local compact rectangles and translate indices at the worker boundary, or own full-frame metadata per active worker and reuse it across scheduled tiles. Keep merge indexing explicit so tile-row and tile-column isolation remains intact. Add an inspectable capacity/allocation regression for a high-column-count frame that proves metadata retention is bounded independently of tile count. Estimated effort: medium.

### Verification summary (2026-10-18T17:30:18Z, deltic:auto run=verify-20261018T172611Z-bacc71dc verified_fix_run=fix-20261016T022804Z-9d7700c8 verdict=close)

Verifier note: Manual verify: The retained decoder worker-count bound test passes; removing the tile-count<=worker-count gate makes its 64-tiles/two-workers assertion fail. Inspection confirms both intra/inter metadata allocation paths follow that common gate and pool conservation keeps retained slab count <= worker count. This proves worker-count eligibility, not a measured capacity or RSS bound; KIL-00080 covers distinct YUV scratch behavior.

## Notes

### 2026-10-16 — existing fix provenance (GPT-6.1 Sol)

The accepted full-frame-per-active-worker alternative already landed in `dc19bcea3f41952f6d0a2a53b027c852a79a5aed`. Its common `parallel_tile_decode_enabled` gate bounds both intra and inter tile counts by the immutable configured worker count before either metadata pool allocates. Pool pop/return accounting preserves that bound across frames and errors. Full-frame capacities still retain historical geometry; no tile-local or measured byte bound is claimed. See [the source and regression evidence](../wrk_journals/2026.10.16%20-%20BUG%20-%20Record%20existing%20decoder%20metadata%20bound.md). State remains Open until the manual landing writer records the verified earlier fixing SHA.

Confirmed by the performance and security lenses, the lead's source trace, and the standing devil's-advocate pass, which set severity to Medium because the review did not quantify an OOM threshold. This is distinct from open `RVP9-BUG-KIL-00080`: that record covers the parallel path's full-frame YUV scratch buffers and their allocation panic, while this record covers segment and MV metadata pools. The July decoder review noted these slabs as report-only; this pass supplied the missing independent verification. No allocation measurement or benchmark was run.
