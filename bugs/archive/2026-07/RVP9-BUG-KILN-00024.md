# RVP9-BUG-KILN-00024 - Direct-to-frame tile and LF workers materialize overlapping full-plane mutable slices

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** core/frame and loopfilter threading aliasing
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T160421Z-p4438-n970866000-c1 branch=task/bug-RVP9-BUG-KILN-00024-run-fix-20260730T160421Z-p4438-n970866000-c1 code=a915c7ff8c204c3527d9c9f38361fa0d96a5edae gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5); residual split to RVP9-BUG-KILN-00062)

## Observation

`TileFrameView::from_yv12` stores full-plane origins and lengths at `crates/rvp9-core/src/frame.rs:1608`, and `TileFrameView::y_plane_mut` reconstructs a full-plane `&mut [u8]` with `from_raw_parts_mut` at `crates/rvp9-core/src/frame.rs:1656`. The decoder constructs one such view per tile from the same backing `Yv12Buffer` at `crates/rvp9-decoder/src/lib.rs:1851` and `crates/rvp9-decoder/src/lib.rs:3169`, then submits tile jobs that pass those views into tile decode at `crates/rvp9-decoder/src/lib.rs:2065` and `crates/rvp9-decoder/src/lib.rs:3426`.

Loop filtering repeats the shape. `SharedFrame::y_slice`, `u_slice`, and `v_slice` return full-plane mutable slices at `crates/rvp9-core/src/loopfilter/apply.rs:159`, and `run_split_edge_lf_worker` materializes all three at `crates/rvp9-core/src/loopfilter/apply.rs:551`. Multiple LF workers are dispatched for the same frame at `crates/rvp9-decoder/src/lib.rs:4235`.

These APIs are trying to enforce disjoint tile or column work by convention, but the actual Rust references cover the full plane ranges. That creates overlapping `&mut` references across workers, and in some cases even overlapping Y/U/V ranges inside one worker because the Y slice starts at `y_offset` and extends through later allocation storage.

## Fix

Open. Avoid materializing full-plane mutable references for split work. Use raw pointers internally until a strictly bounded disjoint slice can be proven, or split the backing buffer into non-overlapping plane/tile ranges before workers run. Add Miri-style aliasing coverage or a focused unsafe-code proof test for the direct-to-frame tile and LF paths.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review. `wrk_docs/2026.05.24 - HLD - Phase3 row-parallel loopfilter.md:474` already records the intended invariant: parallel loop filter workers should operate through raw pointers, not overlapping references.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The unsound construct is REMOVED, not papered over: `TileFrameView::from_yv12`, which handed
every worker a `from_raw_parts_mut` over the whole remaining allocation, is deleted. Workers now
own private `Yv12Buffer`s and only their own mi-rectangle is merged after join
(`crates/rvp9-core/src/frame.rs:208-224`, `:522-663`;
`crates/rvp9-decoder/src/lib.rs:1843-2217`, `:3169-3624`). `y_data_len` was hand-checked to
equal exactly the distance from `y_offset` to the end of the Y plane, so border-extension
writes are not truncated.
`frame::tests::mutable_plane_views_stop_before_the_next_plane` and
`profile1_tile2_parallel_scratch_decodes_bit_exact` both observed passing.
Over-strict-guard check on the new `assert_eq!(num_workers, 1)` in `run_split_edge_lf_worker`
(the RVP9-BUG-KILN-00020 failure mode): all four call sites traced. The live one receives the
literal `1`; two sit behind `OUTPUT_FRAME_LF_OVERLAP_ALIAS_SAFE == false` and are dead; one is
a test. No live caller trips it.
TWO residuals split out, both real. (1) RVP9-BUG-KILN-00062: the third regression test,
`parallel_tile_decode_reuses_isolated_scratch_buffers`, FAILS under the repo's own
`cargo test --workspace` because it samples a process-global allocation counter while sibling
tests decode concurrently -- reproduced deterministically, passes with `--test-threads=1` and in
isolation. (2) Noted not blocking: nothing here proves the ALIASING is gone; this record's own
Fix section asked for Miri-style coverage and none was added -- byte-exact output was never the
failure signal for UB. Also a real perf cost: this reverts direct-to-frame tile decode to
scratch-plus-merge and pins the column-wavefront loop filter to one worker.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
