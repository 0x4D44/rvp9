# RVP9-BUG-FLU-00047 — Production timing matrix exceeds Deltic quiet leases

- **State:** Open
- **Priority:** Must
- **Severity:** Medium
- **Area:** benchmark
- **Raised:** 2026-10-17T13:58:37Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261018T174427Z-66908827
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00047-run-fix-20261018T174427Z-66908827
- **Owner base:** 8873d526c1627864ef28db3d0b78cd7985945ab1
- **Owner fingerprint:** -
- **Owner since:** 2026-10-18T17:44:27Z
- **Owner until:** 2026-10-18T21:44:27Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-17T13:58:37Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium)

## Observation

On the exact fb5c0c917038db6c640b3dbc7c2b84b567233333 candidate on native Windows NMI, the production speed phase requires 75 groups and the recorded host-load guard samples for 60 seconds per group, a 75-minute minimum. Deltic quiet leases are capped at 20 minutes, and the timing phase refuses resume after its first measurement to avoid mixing host states. Expected: the documented complete timing gate can run under the documented quiet policy without combining measurements from different host states. Actual: the complete phase cannot fit in one lease and a partial phase cannot resume; the current record therefore started no timing campaign and has no performance result. See recovered-validation-20261017/postfix-main-qualification/windows-nmi-current-source-pgo-qualification.json (codec_timing) and the production runner timing-resume contract.

## Fix

<unfixed — raised only>

## Notes
