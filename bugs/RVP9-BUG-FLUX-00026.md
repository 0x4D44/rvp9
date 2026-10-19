# RVP9-BUG-FLUX-00026 — GOP topology golden rejects accepted unrestricted-MV output

- **State:** Closed
- **Priority:** Must
- **Severity:** Low
- **Area:** encoder/test infrastructure
- **Raised:** 2026-08-14T23:02:42Z
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
- **State history:** Open (2026-08-14T23:02:42Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-14T23:08:58Z, deltic:auto role=fix run=fix-20260814T230323Z-p28077-n811046000-c1 branch=task/bug-RVP9-BUG-FLUX-00026-run-fix-20260814T230323Z-p28077-n811046000-c1 code=982de73e gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260814T230323Z-p28077-n811046000-c1 branch=task/bug-RVP9-BUG-FLUX-00026-run-fix-20260814T230323Z-p28077-n811046000-c1 code=93653fb2e77c0e207fef344a334bad556efc232f gate=manual provenance-only=true reconciled-by=Codex) -> Closed (2026-10-19T16:13:05Z, deltic:manual role=verify run=verify-20261019T160745Z-f45c75cd verified_fix_run=fix-20260814T230323Z-p28077-n811046000-c1 verdict=close)

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

### Verification summary (2026-10-19T16:13:05Z, deltic:auto run=verify-20261019T160745Z-f45c75cd verified_fix_run=fix-20260814T230323Z-p28077-n811046000-c1 verdict=close)

Verifier note: Manual verify: Current origin/main release run passed both GOP tests at 22914 bytes / 0xea3e42753795d112. On fix commit 93653fb2, restored only the pre-fix golden (24229 / 0x124b0ed76b9e627d); the selected gop_topology_byte_identity test failed its length assertion with 22914 vs 24229. Restored the accepted constants, confirmed an empty scratch diff, and reran both release tests: 2 passed. The schedule-and-decode oracle remains green.

## Notes
