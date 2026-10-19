# RVP9-BUG-FLU-00030 — Timing qualifiers cannot enforce the idle-host gate on Windows

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** benchmark
- **Raised:** 2026-08-22T18:08:12Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T111120Z-cb2860ec
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00030-run-verify-20261019T111120Z-cb2860ec
- **Owner base:** e52e70d25cda9218ec8dd0db906e06ad4b444ad1
- **Owner fingerprint:** sha256:aaa663ba54ac9f64c772ab7ba120dec1149c6343079dcbc927949746d8b6eb1b
- **Owner since:** 2026-10-19T11:11:20Z
- **Owner until:** 2026-10-19T15:11:20Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T18:08:12Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-22T18:20:49Z, deltic:auto role=fix run=fix-20260822T180840Z-795f80a2 branch=task/bug-RVP9-BUG-FLU-00030-run-fix-20260822T180840Z-795f80a2 code=36eec127 gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260822T180840Z-795f80a2 branch=task/bug-RVP9-BUG-FLU-00030-run-fix-20260822T180840Z-795f80a2 code=1d5a1dfea4f6a52aa755c71774db17ecb3bda03a gate=manual provenance-only=true reconciled-by=Codex)

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

## Notes
