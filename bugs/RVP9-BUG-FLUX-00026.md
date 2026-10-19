# RVP9-BUG-FLUX-00026 — GOP topology golden rejects accepted unrestricted-MV output

- **State:** Fixed
- **Priority:** Must
- **Severity:** Low
- **Area:** encoder/test infrastructure
- **Raised:** 2026-08-14T23:02:42Z
- **Discovery source:** Automation
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T114949Z-c950098b
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00026-run-verify-20261019T114949Z-c950098b
- **Owner base:** bec32f7923d752285d81b6aed9fa76b2fe5e7d5b
- **Owner fingerprint:** sha256:2c9778c877ff7d0ce502edf81a508d8c4183935002a9ef62f7065cb2f2c693b3
- **Owner since:** 2026-10-19T11:49:49Z
- **Owner until:** 2026-10-19T15:49:49Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T23:02:42Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-14T23:08:58Z, deltic:auto role=fix run=fix-20260814T230323Z-p28077-n811046000-c1 branch=task/bug-RVP9-BUG-FLUX-00026-run-fix-20260814T230323Z-p28077-n811046000-c1 code=982de73e gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260814T230323Z-p28077-n811046000-c1 branch=task/bug-RVP9-BUG-FLUX-00026-run-fix-20260814T230323Z-p28077-n811046000-c1 code=93653fb2e77c0e207fef344a334bad556efc232f gate=manual provenance-only=true reconciled-by=Codex)

## Observation

On the integrated release candidate, cargo test -p rvp9-encoder --release --locked --test w_gop_topology fails the byte golden: current output is 22,914 bytes with FNV 0xea3e42753795d112, versus the stale expected 24,229 bytes and 0x124b0ed76b9e627d. The sibling schedule-and-decode oracle passes. An isolated bisect identifies e5a57f0a98ff25d04e1986177f7383cbce20ef6a as the first changer, and that first-bad commit produces exactly the current bytes, tying the change to the accepted unrestricted-motion policy rather than a later regression.

## Fix

Historical fix provenance is reconciled below. Independent behavioral verification remains pending.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `93653fb2e77c0e207fef344a334bad556efc232f` (test(enc): refresh GOP golden after UMV policy).
The original reference `982de73e` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `a7ace4f85d9fc6ae1a0508c8529bdc37155efe3d` has this fixing commit as its immediate parent.
All original old/new length and FNV constants match exact candidate patch; commit names bug, same unchanged sibling oracle and UMV first-changer explanation. Release journal also records full missing982de73e630d98f575871b7666eccebdcbd1e4dc.
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

## Notes
