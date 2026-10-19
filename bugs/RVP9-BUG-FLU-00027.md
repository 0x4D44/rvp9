# RVP9-BUG-FLU-00027 — Loop-filter SharedFrame invalidates plane pointer provenance

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** core
- **Raised:** 2026-08-20T20:38:47Z
- **Discovery source:** Automation
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T070437Z-b3385cd0
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00027-run-verify-20261019T070437Z-b3385cd0
- **Owner base:** 76390f1f0e067d3755e3aa28834ef2837f70bf93
- **Owner fingerprint:** sha256:c191bbfbe1cc5c51f99e0a31a9539040990376f7e8c1da9bfd9a0024b14850c8
- **Owner since:** 2026-10-19T07:04:37Z
- **Owner until:** 2026-10-19T11:04:37Z
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
