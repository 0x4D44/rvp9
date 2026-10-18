# RVP9-BUG-FLU-00057 — Spatial speed summary emits unescaped clip lists into CSV columns

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** qualification/spatial-summary
- **Raised:** 2026-10-18T20:10:46Z
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
- **State history:** Open (2026-10-18T20:10:46Z, raised via `deltic bugs new --land` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T20:13:20Z, deltic:auto role=fix run=fix-20261018T201105Z-bbaaf303 branch=task/bug-RVP9-BUG-FLU-00057-run-fix-20261018T201105Z-bbaaf303 code=d44c4b28987ad97da05c412267d9725b0e5fcfe1 gate=manual)

## Observation

The spatial speed phase builds clips as a comma-separated list in run_speed_group, then interpolates it unquoted into speed-summary.csv. The header declares 16 columns, but a representative 720p30 row with three clips parses as 18 fields (and two-clip targets produce 17). Expected: each data row matches the header and keeps the full clips list in one CSV field. Evidence: scripts/qualify-spatial-svc.sh lines 1101-1104, 1154-1160, and 1173-1174; Python csv.reader confirmed 16 header fields and 18 representative row fields.

Evidence fingerprint: `manual:v1:spatial-speed-summary-emits-unescaped-clip-list-99464d903f7e88a5`


## Fix

The spatial speed summary now writes the comma-separated clip list as a quoted `clips` field. `run_speed_group` uses the tested `format_speed_summary_row` formatter, and the no-I/O script self-test checks a representative three-clip row.

Validation: before the fix, the new self-test failed only the clip-field assertion and showed the three clip names split into separate CSV columns. After the fix, `scripts/qualify-spatial-svc.sh --self-test`, `bash -n scripts/qualify-spatial-svc.sh`, a Python `csv.reader` check for 16 columns with the full clip list in column 6, and `git diff --check` passed.

## Notes
