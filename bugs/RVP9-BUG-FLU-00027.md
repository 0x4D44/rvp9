# RVP9-BUG-FLU-00027 — Loop-filter SharedFrame invalidates plane pointer provenance

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** core
- **Raised:** 2026-08-20T20:38:47Z
- **Discovery source:** Automation
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
- **State history:** Open (2026-08-20T20:38:47Z, raised via `deltic bugs new` model=gpt-5.6@xhigh) -> Fixed (2026-08-21T22:18:40Z, deltic:auto role=fix run=fix-20260821T221759Z-56a3c2e3 branch=task/bug-RVP9-BUG-FLU-00027-run-fix-20260821T221759Z-56a3c2e3 code=3196ff31 gate=manual)

## Observation

On AArch64 macOS, MIRIFLAGS=-Zmiri-disable-isolation cargo miri test --package rvp9-core --lib loopfilter::apply::tests::empty_mask_grid_no_op reports Undefined Behavior at SharedFrame::y_slice: the Y pointer tag was invalidated by a later unique retag in Yv12Buffer::u_plane_mut. Expected: the empty-mask loop-filter path completes without an aliasing violation. Actual: Miri aborts while reconstructing the Y slice.

## Fix

<unfixed — raised only>

## Notes
