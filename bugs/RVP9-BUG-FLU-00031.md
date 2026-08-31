# RVP9-BUG-FLU-00031 — Windows load sampling discards benchmarks on transient PDH failures

- **State:** Fixed
- **Priority:** Must
- **Severity:** Medium
- **Area:** bench/encode-headtohead
- **Raised:** 2026-08-31T15:05:29Z
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
- **State history:** Open (2026-08-31T15:05:29Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-31T15:12:46Z, deltic:auto role=fix run=fix-20260831T150649Z-e1f6214c branch=task/bug-RVP9-BUG-FLU-00031-run-fix-20260831T150649Z-e1f6214c code=b437da2e gate=manual)

## Observation

On Crucible, encode-headtohead twice aborted before timing with cannot read host one-minute load when Get-Counter transiently failed, after the same harness completed eight load samples in an A/A run. Expected: a transient Windows PDH read failure is retried while the load gate remains fail-closed if all bounded attempts fail. Actual: one failed 60-second sample discards the entire benchmark attempt and produces no summary.csv.

## Fix

<unfixed — raised only>

## Notes
