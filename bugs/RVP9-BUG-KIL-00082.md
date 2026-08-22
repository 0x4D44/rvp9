# RVP9-BUG-KIL-00082 — Parallel decoder retains full-frame metadata slabs per tile

- **State:** Open
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
- **State history:** Open (2026-08-22T10:24:29Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

The parallel intra path creates one `Vec<u8>` per tile at `crates/rvp9-decoder/src/lib.rs:1947-1963`, and every vector is sized to `mi_rows * mi_cols`. The merge reads only the tile rectangle at `:2280-2295`. The parallel inter path repeats that full-frame segment map and also creates one full-frame `Vec<PrevFrameMvRef>` per tile at `:3300-3331`; its merge likewise copies only the tile rectangle at `:3716-3732`.

Both pools persist on `Decoder` at `:555-570`, so the maximum tile-count allocation stays retained after the frame. Valid multi-column input therefore makes metadata memory and cold allocation scale as `O(tile_count * frame_area)`, although only the bounded active workers run concurrently and each tile accesses only its own rectangle.

Expected: worker metadata should scale with tile-local area or the number of active workers. Actual: the decoder retains one frame-wide segment slab per tile and, on inter frames, one frame-wide MV slab per tile.

## Fix

Store tile-local compact rectangles and translate indices at the worker boundary, or own full-frame metadata per active worker and reuse it across scheduled tiles. Keep merge indexing explicit so tile-row and tile-column isolation remains intact. Add an inspectable capacity/allocation regression for a high-column-count frame that proves metadata retention is bounded independently of tile count. Estimated effort: medium.

## Notes

Confirmed by the performance and security lenses, the lead's source trace, and the standing devil's-advocate pass, which set severity to Medium because the review did not quantify an OOM threshold. This is distinct from open `RVP9-BUG-KIL-00080`: that record covers the parallel path's full-frame YUV scratch buffers and their allocation panic, while this record covers segment and MV metadata pools. The July decoder review noted these slabs as report-only; this pass supplied the missing independent verification. No allocation measurement or benchmark was run.
