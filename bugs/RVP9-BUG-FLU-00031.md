# RVP9-BUG-FLU-00031 — Windows load sampling discards benchmarks on transient PDH failures

- **State:** Closed
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
- **State history:** Open (2026-08-31T15:05:29Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-31T15:12:46Z, deltic:auto role=fix run=fix-20260831T150649Z-e1f6214c branch=task/bug-RVP9-BUG-FLU-00031-run-fix-20260831T150649Z-e1f6214c code=b437da2e gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260831T150649Z-e1f6214c branch=task/bug-RVP9-BUG-FLU-00031-run-fix-20260831T150649Z-e1f6214c code=ba67e87bc0bb316390cf4e1b07cd6ad78bd829cc gate=manual provenance-only=true reconciled-by=Codex) -> Closed (2026-10-19T11:34:29Z, deltic:manual role=verify run=verify-20261019T113101Z-eaebee0b verified_fix_run=fix-20260831T150649Z-e1f6214c verdict=close)

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

### Verification summary (2026-10-19T11:34:29Z, deltic:auto run=verify-20261019T113101Z-eaebee0b verified_fix_run=fix-20260831T150649Z-e1f6214c verdict=close)

Verifier note: Manual verify: Independent verification on macOS at the probe boundary: cargo test -p rvp9-bench --bin encode-headtohead load_ --locked passed 6/6; guarded enforcement retries once and then fails closed after two failures. Fails-before: changed load_with_retry to a single read; recovery failed with None instead of Some(0.5), the two-failure call count was 1 instead of 2, and guarded_failures was 1 instead of 2. Restored read().or_else(read), confirmed the source diff was empty, and reran 6/6 green. This verifies the transient probe-failure behavior; native Windows Get-Counter/PDH was not exerc...

## Notes
