# RVP9-BUG-FLU-00027 — Loop-filter SharedFrame invalidates plane pointer provenance

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** core
- **Raised:** 2026-08-20T20:38:47Z
- **Discovery source:** Automation
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260821T221759Z-56a3c2e3
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00027-run-fix-20260821T221759Z-56a3c2e3
- **Owner base:** cb2a82deda50e9ce5fc6934ff3db933b732455f9
- **Owner fingerprint:** -
- **Owner since:** 2026-08-21T22:17:59Z
- **Owner until:** 2026-08-22T00:17:59Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-20T20:38:47Z, raised via `deltic bugs new` model=gpt-5.6@xhigh)

## Observation

On AArch64 macOS, MIRIFLAGS=-Zmiri-disable-isolation cargo miri test --package rvp9-core --lib loopfilter::apply::tests::empty_mask_grid_no_op reports Undefined Behavior at SharedFrame::y_slice: the Y pointer tag was invalidated by a later unique retag in Yv12Buffer::u_plane_mut. Expected: the empty-mask loop-filter path completes without an aliasing violation. Actual: Miri aborts while reconstructing the Y slice.

## Fix

<unfixed — raised only>

## Notes
