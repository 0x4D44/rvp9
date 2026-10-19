# RVP9-BUG-FLU-00031 — Windows load sampling discards benchmarks on transient PDH failures

- **State:** Fixed
- **Priority:** Must
- **Severity:** Medium
- **Area:** bench/encode-headtohead
- **Raised:** 2026-08-31T15:05:29Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T113101Z-eaebee0b
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00031-run-verify-20261019T113101Z-eaebee0b
- **Owner base:** d5240272fb6c9064a833e231fccc4d85c58c2c0f
- **Owner fingerprint:** sha256:45f3f1415732057899b7b92aec4d37c68f47fd2b243846d7cdb1f7f1e13f57bb
- **Owner since:** 2026-10-19T11:31:01Z
- **Owner until:** 2026-10-19T15:31:01Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-31T15:05:29Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-31T15:12:46Z, deltic:auto role=fix run=fix-20260831T150649Z-e1f6214c branch=task/bug-RVP9-BUG-FLU-00031-run-fix-20260831T150649Z-e1f6214c code=b437da2e gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260831T150649Z-e1f6214c branch=task/bug-RVP9-BUG-FLU-00031-run-fix-20260831T150649Z-e1f6214c code=ba67e87bc0bb316390cf4e1b07cd6ad78bd829cc gate=manual provenance-only=true reconciled-by=Codex)

## Observation

On Crucible, encode-headtohead twice aborted before timing with cannot read host one-minute load when Get-Counter transiently failed, after the same harness completed eight load samples in an A/A run. Expected: a transient Windows PDH read failure is retried while the load gate remains fail-closed if all bounded attempts fail. Actual: one failed 60-second sample discards the entire benchmark attempt and produces no summary.csv.

## Fix

Historical fix provenance is reconciled below. Independent behavioral verification remains pending.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `ba67e87bc0bb316390cf4e1b07cd6ad78bd829cc` (fix(bench): retry transient Windows load reads).
The original reference `b437da2e` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `615c1245f577d94b1b64e60667cc1aeab59c4f49` has this fixing commit as its immediate parent.
Commit message explicitly names RVP9-BUG-FLU-00031. Adds load_with_retry using read().or_else(read), replaces enforce_load direct current_load1 with bounded retry, and retains fail-closed error. Adds three named regressions for recovery after one failure, failure after two reads, and no repeated valid read. Exact recorded transient PDH retry repair.
Contemporary corroboration: wrk_journals/2026.07.30 - JRN - production parity drive.md:36888 (explicit bug31 and old b437da2e bounded-retry landed reference).
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

## Notes
