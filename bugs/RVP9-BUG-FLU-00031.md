# RVP9-BUG-FLU-00031 — Windows load sampling discards benchmarks on transient PDH failures

- **State:** Fixed
- **Priority:** Must
- **Severity:** Medium
- **Area:** bench/encode-headtohead
- **Raised:** 2026-08-31T15:05:29Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T070528Z-5882044e
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00031-run-verify-20261019T070528Z-5882044e
- **Owner base:** 6c7ee85704402b690bc98197537acf3381f6af7c
- **Owner fingerprint:** sha256:bdc60a638f303810982c4b27d77f0d591b820cb72dca758dfd97f3335b30c338
- **Owner since:** 2026-10-19T07:05:28Z
- **Owner until:** 2026-10-19T11:05:28Z
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
