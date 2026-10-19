# RVP9-BUG-FLU-00030 — Timing qualifiers cannot enforce the idle-host gate on Windows

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** benchmark
- **Raised:** 2026-08-22T18:08:12Z
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
- **State history:** Open (2026-08-22T18:08:12Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-22T18:20:49Z, deltic:auto role=fix run=fix-20260822T180840Z-795f80a2 branch=task/bug-RVP9-BUG-FLU-00030-run-fix-20260822T180840Z-795f80a2 code=36eec127 gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260822T180840Z-795f80a2 branch=task/bug-RVP9-BUG-FLU-00030-run-fix-20260822T180840Z-795f80a2 code=1d5a1dfea4f6a52aa755c71774db17ecb3bda03a gate=manual provenance-only=true reconciled-by=Codex) -> Closed (2026-10-19T11:30:17Z, deltic:manual role=verify run=verify-20261019T111120Z-cb2860ec verified_fix_run=fix-20260822T180840Z-795f80a2 verdict=close)

## Observation

On an otherwise idle Windows x86_64 host, qualify-spatial-svc.sh and qualify-production-encoder.sh cannot read /proc/loadavg or vm.loadavg, while encode-headtohead has the same Unix-only current_load1 implementation. Every timing phase therefore fails closed before measuring. Expected: the checked-in acceptance qualifier should measure a conservative Windows host-idle equivalent and preserve the 0.75 busy-core ceiling.

## Fix

Historical fix provenance is reconciled below. Independent behavioral verification remains pending.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `1d5a1dfea4f6a52aa755c71774db17ecb3bda03a` (fix(bench): measure Windows acceptance host load).
The original reference `36eec127` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `698937ed3bd2592f0ab4061919150fb1a20410df` has this fixing commit as its immediate parent.
All three named affected files gain native Windows host-load sampling: PowerShell Get-Counter, thirteen five-second samples dropping baseline, mean percent normalized by ProcessorCount to busy logical cores. Rust adds cfg(windows) current_load1; both qualifiers route MINGW/MSYS/CYGWIN through windows_load1 before synthetic Unix loadavg. Existing ceiling remains unchanged and metric enters manifests. Exact recorded Windows idle-gate repair.
Contemporary corroboration: wrk_docs/2026.08.23 - DAILY - KILN-Windows - Repo state report.md:99,123 (bug30 old36eec127 pairing and Windows timing idle-host guard description).
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

### Verification summary (2026-10-19T11:30:17Z, deltic:auto run=verify-20261019T111120Z-cb2860ec verified_fix_run=fix-20260822T180840Z-795f80a2 verdict=close)

Verifier note: Manual verify: Manual verify on origin fix 1d5a1dfea4f6a52aa755c71774db17ecb3bda03a (2026-10-19). Windows 11 x64 / PowerShell 5.1: cargo test -p rvp9-bench --bin encode-headtohead --locked tests::windows_idle_guard_produces_a_normalized_load_measurement -- --exact selected 1 and passed. Fails-before: replaced only the Windows current_load1 body in registered remote scratch with the pre-fix /proc/loadavg + sysctl probe; the same selected test failed at “Windows host load should be measurable”. Re-extracted tracked HEAD (SHA-256 restored to 62076be9c6bc65d552d896163cd687db48a8c0dc65f3716526415db...

## Notes
