# RVP9-BUG-FLUX-00026 — GOP topology golden rejects accepted unrestricted-MV output

- **State:** Fixed
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
- **State history:** Open (2026-08-14T23:02:42Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-14T23:08:58Z, deltic:auto role=fix run=fix-20260814T230323Z-p28077-n811046000-c1 branch=task/bug-RVP9-BUG-FLUX-00026-run-fix-20260814T230323Z-p28077-n811046000-c1 code=982de73e gate=manual)

## Observation

On the integrated release candidate, cargo test -p rvp9-encoder --release --locked --test w_gop_topology fails the byte golden: current output is 22,914 bytes with FNV 0xea3e42753795d112, versus the stale expected 24,229 bytes and 0x124b0ed76b9e627d. The sibling schedule-and-decode oracle passes. An isolated bisect identifies e5a57f0a98ff25d04e1986177f7383cbce20ef6a as the first changer, and that first-bad commit produces exactly the current bytes, tying the change to the accepted unrestricted-motion policy rather than a later regression.

## Fix

<unfixed — raised only>

## Notes
