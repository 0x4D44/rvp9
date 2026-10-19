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
- **State history:** Open (2026-08-20T20:38:47Z, raised via `deltic bugs new` model=gpt-5.6@xhigh) -> Fixed (2026-08-21T22:18:40Z, deltic:auto role=fix run=fix-20260821T221759Z-56a3c2e3 branch=task/bug-RVP9-BUG-FLU-00027-run-fix-20260821T221759Z-56a3c2e3 code=3196ff31 gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260821T221759Z-56a3c2e3 branch=task/bug-RVP9-BUG-FLU-00027-run-fix-20260821T221759Z-56a3c2e3 code=34f273fe8b95013bcca01224ec0dbd491b107054 gate=manual provenance-only=true reconciled-by=Codex)

## Observation

On AArch64 macOS, MIRIFLAGS=-Zmiri-disable-isolation cargo miri test --package rvp9-core --lib loopfilter::apply::tests::empty_mask_grid_no_op reports Undefined Behavior at SharedFrame::y_slice: the Y pointer tag was invalidated by a later unique retag in Yv12Buffer::u_plane_mut. Expected: the empty-mask loop-filter path completes without an aliasing violation. Actual: Miri aborts while reconstructing the Y slice.

## Fix

Historical fix provenance is reconciled below. Independent behavioral verification remains pending.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `34f273fe8b95013bcca01224ec0dbd491b107054` (fix(core): preserve loop-filter plane provenance).
The original reference `3196ff31` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `c7a9d7e2c7ff16a4fef94bd080cfce1bef31cbfb` contains the original reference and descends from this fixing commit.
Commit explicitly says Fixes: RVP9-BUG-FLU-00027. frame.rs adds Yv12Buffer::yuv_planes_mut using one split_at_mut chain; SharedFrame::from_yv12 replaces sequential y/u/v_plane_mut calls with simultaneously live split plane views. This is the exact original pointer-provenance repair.
Contemporary corroboration: wrk_journals/2026.07.30 - JRN - production parity drive.md:36258 (Unit 1254; original Miri finding and disjoint split repair; journal added in this commit); wrk_docs/2026.08.22 - DAILY - KILN-Windows - Repo state report.md:89 (old reference and bug pairing).
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

## Notes
