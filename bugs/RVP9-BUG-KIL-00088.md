# RVP9-BUG-KIL-00088 — dump-frame-diff panics at small and partial frame edges

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** harness/dump-frame-diff
- **Raised:** 2026-08-22T13:20:53Z
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
- **State history:** Open (2026-08-22T13:20:53Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T08:40:30Z, deltic:auto role=fix run=fix-20261016T083030Z-646473d7 branch=task/bug-RVP9-BUG-KIL-00088-run-fix-20261016T083030Z-646473d7 code=bb7a759f5fb20bc7bd05f83a3e0af0fdc8c6556a gate=manual) -> Closed (2026-10-18T17:25:23Z, deltic:manual role=verify run=verify-20261018T172231Z-523935bf verified_fix_run=fix-20261016T083030Z-646473d7 verdict=close)

## Observation

`dump-frame-diff` unconditionally reads a 12-column by 8-row luma window at
`crates/rvp9-harness/src/bin/dump_frame_diff.rs:219-230`. Any otherwise valid
decoded frame smaller than that window panics. The block grid also uses
ceiling division, but the first-divergent-block dump at lines 290-324 always
indexes a full 4x4 block. A mismatch in a partial right or bottom edge block of
a frame whose visible dimensions are not multiples of four can therefore index
past the visible-plane buffer and panic instead of printing the diagnostic.

Expected: the diagnostic clips each dump to visible dimensions. Actual: valid
small or edge-aligned inputs can crash exactly when a divergence is found.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T17:25:23Z, deltic:auto run=verify-20261018T172231Z-523935bf verified_fix_run=fix-20261016T083030Z-646473d7 verdict=close)

Verifier note: Manual verify: The two retained dump-frame-diff regressions pass on the fixed build. Restoring the unbounded 12x8 window and unconditional 4x4 difference blocks makes the small/partial-frame tests fail with bounds panics and row-wrapping mismatches. Exact source restored. Coverage uses synthetic visible-plane fixtures on ARM64, not an end-to-end corpus decode.

## Notes

Bound the fixed window by `crop_w`/`crop_h` and bound each partial block's row
and column loops. Add small-dimension and non-multiple-of-four unit fixtures.
Confirmed statically; no decoder run was performed.
