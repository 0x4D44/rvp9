# RVP9-BUG-KIL-00078 — Tile-info writer accepts invalid geometry in release builds

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/bitstream header writer
- **Raised:** 2026-08-22T08:56:49Z
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
- **State history:** Open (2026-08-22T08:56:49Z, raised via `deltic bugs new`) -> Fixed (2026-08-22T13:26:41Z, deltic:auto role=fix run=fix-20260822T132113Z-b15a1b7a branch=task/bug-RVP9-BUG-KIL-00078-run-fix-20260822T132113Z-b15a1b7a code=bdb22b49 gate=manual) -> Closed (2026-10-18T16:50:15Z, deltic:manual role=verify run=verify-20261018T163531Z-60b355f1 verified_fix_run=fix-20260822T132113Z-b15a1b7a verdict=close)

## Observation

The public `write_tile_info` helper returns `Result<(), BitstreamError>`, but it enforces the frame-dependent `log2_tile_cols` range only with `debug_assert!` at `crates/rvp9-core/src/bitstream/sub_parsers.rs:569-579`. Release builds then subtract and emit the unchecked value at `:581-590`. A value above the computed maximum writes excess increment bits, shifting the row and following header fields; a value below the minimum is silently encoded as the minimum. `log2_tile_rows` is also unchecked at `:592-597`, so every value above 2 is silently encoded as 2.

Expected: invalid public `TileInfo` values produce a typed error consistently in debug and release builds. Actual: debug builds panic for invalid columns while release builds return `Ok(())` after emitting a header that represents different tile geometry or shifts subsequent fields. `write_uncompressed_header` calls this helper at `crates/rvp9-core/src/bitstream/header.rs:717-721`, so the defect is also exposed through the public header writer.

## Proposed resolution

Validate `log2_tile_cols` against the computed inclusive range and reject `log2_tile_rows > 2` before writing any bits. Return a specific `BitstreamError`, use checked geometry for invalid `mi_cols`, and add regressions for below-minimum, above-maximum, row value 3, and release-equivalent behavior. Estimated effort: small.

## Review evidence

Confirmed independently by the reliability reviewer, the lead's source trace, and the standing devil's advocate. This is distinct from closed RVP9-BUG-KILN-00023, which covered tile payload-buffer dimensions and counts rather than the tile-info header writer. No code was executed in this review pass.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T16:50:15Z, deltic:auto run=verify-20261018T163531Z-60b355f1 verified_fix_run=fix-20260822T132113Z-b15a1b7a verdict=close)

Verifier note: Manual verify: Independent verification passed both retained release tests on the fixed build. Restoring debug-only column validation and unchecked rows makes the selected regression fail because invalid geometry returns Ok. Release public-header probes return typed InvalidTileInfo errors for below-minimum columns, above-maximum columns and rows=3; the helper emits no tile bits, while the streaming header writer has already emitted its 95-bit prefix. The final duplicate Cargo rebuild timed out at 300s; source was restored byte-for-byte and the same retained tests passed in the optimized assert...

## Notes
