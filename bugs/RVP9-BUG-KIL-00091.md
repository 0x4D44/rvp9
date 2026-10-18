# RVP9-BUG-KIL-00091 — check-lld-format can omit unreadable inputs and exit successfully

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** harness/check-lld-format
- **Raised:** 2026-08-22T13:20:53Z
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
- **State history:** Open (2026-08-22T13:20:53Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T06:16:50Z, deltic:auto role=fix run=fix-20261016T060707Z-0e7ce44a branch=task/bug-RVP9-BUG-KIL-00091-run-fix-20261016T060707Z-0e7ce44a code=225595e6c1c3dd3c190c23bdba4edeb9a7719c0d gate=manual) -> Closed (2026-10-18T15:42:23Z, deltic:manual role=verify run=verify-20261018T153844Z-879c82c8 verified_fix_run=fix-20261016T060707Z-0e7ce44a verdict=close)

## Observation

`check-lld-format` silently drops directory-entry errors via
`entries.flatten()` and treats a per-file `check_one` read failure as a warning
at `crates/rvp9-harness/src/bin/check_lld_format.rs:195-209`. The final failure
flag at lines 224-257 is computed only from successfully read reports. It can
therefore exit zero after omitting one or even every candidate LLD.

Expected: every discovered input is accounted for, and an unreadable input
makes the checker fail. Actual: the command can report that all checked files
are structurally sound while silently excluding files it could not inspect.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T15:42:23Z, deltic:auto run=verify-20261018T153844Z-879c82c8 verified_fix_run=fix-20261016T060707Z-0e7ce44a verdict=close)

Verifier note: Manual verify: Independent session01a14fa5, not fixer. Base09697b55 contains225595e6. Original omitted-input observations exercised: invalid UTF8 candidate alone/mixed and injected directory-entry error. Bin6+CLI6 tests pass before/after reversal. Mutant exit flag=false: both unreadable CLI tests fail status0 vs1. Mutant Err=>continue: injected-entry test fails count0 vs1. Exact source restored, scoped diff empty. Root errors now feed input_failures and exit1 while valid report remains. Workspace fmt and documented clippy pass (warnings retained). Full evidence in 2026.10.18 independent verifi...

## Notes

Treat directory iteration and file-read errors as input failures and return a
non-zero exit. Add an injected/read-error regression at the directory-walk
boundary. Confirmed statically; no checker run was performed.
