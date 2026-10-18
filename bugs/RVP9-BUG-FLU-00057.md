# RVP9-BUG-FLU-00057 — Spatial speed summary emits unescaped clip lists into CSV columns

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** qualification/spatial-summary
- **Raised:** 2026-10-18T20:10:46Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261018T201105Z-bbaaf303
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00057-run-fix-20261018T201105Z-bbaaf303
- **Owner base:** c9db37af02b35bdc9d73aaadbf7e65385f16463c
- **Owner fingerprint:** -
- **Owner since:** 2026-10-18T20:11:05Z
- **Owner until:** 2026-10-19T00:11:05Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-18T20:10:46Z, raised via `deltic bugs new --land` model=gpt-6.1-sol@medium)

## Observation

The spatial speed phase builds clips as a comma-separated list in run_speed_group, then interpolates it unquoted into speed-summary.csv. The header declares 16 columns, but a representative 720p30 row with three clips parses as 18 fields (and two-clip targets produce 17). Expected: each data row matches the header and keeps the full clips list in one CSV field. Evidence: scripts/qualify-spatial-svc.sh lines 1101-1104, 1154-1160, and 1173-1174; Python csv.reader confirmed 16 header fields and 18 representative row fields.

Evidence fingerprint: `manual:v1:spatial-speed-summary-emits-unescaped-clip-list-99464d903f7e88a5`


## Fix

<unfixed — raised only>

## Notes
