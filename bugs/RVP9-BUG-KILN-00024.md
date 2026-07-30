# RVP9-BUG-KILN-00024 - Direct-to-frame tile and LF workers materialize overlapping full-plane mutable slices

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** core/frame and loopfilter threading aliasing
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T160421Z-p4438-n970866000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00024-run-fix-20260730T160421Z-p4438-n970866000-c1
- **Owner base:** 0dc86dac34d685b22a09be8381d8d49d505baba6
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T16:04:21Z
- **Owner until:** 2026-07-30T18:04:21Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`)

## Observation

`TileFrameView::from_yv12` stores full-plane origins and lengths at `crates/rvp9-core/src/frame.rs:1608`, and `TileFrameView::y_plane_mut` reconstructs a full-plane `&mut [u8]` with `from_raw_parts_mut` at `crates/rvp9-core/src/frame.rs:1656`. The decoder constructs one such view per tile from the same backing `Yv12Buffer` at `crates/rvp9-decoder/src/lib.rs:1851` and `crates/rvp9-decoder/src/lib.rs:3169`, then submits tile jobs that pass those views into tile decode at `crates/rvp9-decoder/src/lib.rs:2065` and `crates/rvp9-decoder/src/lib.rs:3426`.

Loop filtering repeats the shape. `SharedFrame::y_slice`, `u_slice`, and `v_slice` return full-plane mutable slices at `crates/rvp9-core/src/loopfilter/apply.rs:159`, and `run_split_edge_lf_worker` materializes all three at `crates/rvp9-core/src/loopfilter/apply.rs:551`. Multiple LF workers are dispatched for the same frame at `crates/rvp9-decoder/src/lib.rs:4235`.

These APIs are trying to enforce disjoint tile or column work by convention, but the actual Rust references cover the full plane ranges. That creates overlapping `&mut` references across workers, and in some cases even overlapping Y/U/V ranges inside one worker because the Y slice starts at `y_offset` and extends through later allocation storage.

## Fix

Open. Avoid materializing full-plane mutable references for split work. Use raw pointers internally until a strictly bounded disjoint slice can be proven, or split the backing buffer into non-overlapping plane/tile ranges before workers run. Add Miri-style aliasing coverage or a focused unsafe-code proof test for the direct-to-frame tile and LF paths.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review. `wrk_docs/2026.05.24 - HLD - Phase3 row-parallel loopfilter.md:474` already records the intended invariant: parallel loop filter workers should operate through raw pointers, not overlapping references.
