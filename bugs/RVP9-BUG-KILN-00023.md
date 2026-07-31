# RVP9-BUG-KILN-00023 - Public tile buffer helpers trust invalid tile counts

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T175217Z-p93805-n268059000-c1 branch=task/bug-RVP9-BUG-KILN-00023-run-fix-20260730T175217Z-p93805-n268059000-c1 code=17329ce37d02956760b50a98b1c039a189e8dd0f gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`crates/rvp9-core/src/bitstream/tile.rs:24` documents the VP9 tile limit, but public `parse_tile_buffers` enforces only zero-dimension rejection before computing `tile_cols * tile_rows` and allocating `Vec::with_capacity(total)` at `crates/rvp9-core/src/bitstream/tile.rs:70`. Its docs claim no panic at `crates/rvp9-core/src/bitstream/tile.rs:58`, but hostile public callers can still request unbounded helper allocation.

The paired `write_tile_buffers` helper computes the expected payload count at `crates/rvp9-core/src/bitstream/tile.rs:145`, but checks it only with `debug_assert_eq!` at `crates/rvp9-core/src/bitstream/tile.rs:146`. Release builds emit data based on the actual payload list at `crates/rvp9-core/src/bitstream/tile.rs:156`, even when that count does not match the supplied tile geometry.

## Fix

Open. Enforce VP9 tile bounds and payload counts in release code at the public helper boundary. Return typed errors for invalid tile dimensions or mismatched payload counts, and add tests for huge tile dimensions plus mismatched write counts.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review. Normal decoder header parsing appears to clamp tile counts before this helper, so the confirmed product impact is the public helper contract rather than the current decoder malformed-stream path.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Both halves of the observation are addressed at the right layer: the tile grid is bounded to
VP9's 1..=64 x 1..=4 before `Vec::with_capacity` (`crates/rvp9-core/src/bitstream/tile.rs:32-42`,
used at `:81` and `:160`), and the writer's `debug_assert_eq!` became a real
`TilePayloadCountMismatch` error that fires in release.
`zero_tile_dims_are_rejected`, `tile_dimensions_beyond_vp9_limits_are_rejected`,
`write_tile_buffers_zero_dimensions_are_rejected` and
`write_tile_buffers_rejects_payload_count_mismatch` all observed passing.
Over-strict-guard check: `parse_tile_buffers(buf, 0, n)` used to return `Ok(vec![])` and now
returns `Err`. Both in-tree callers (`crates/rvp9-decoder/src/lib.rs:1639`, `:2858`) compute
`1 << log2_tile_cols` / `1 << log2_tile_rows`, capped at 6 and 2 bits respectively, so both are
always in range. No real caller hits the new rejection.
Residual noted, not blocking: `write_tile_buffers` has no non-test caller, so its correctness
rests on the unit test alone.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
