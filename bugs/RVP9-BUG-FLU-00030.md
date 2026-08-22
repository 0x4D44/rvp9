# RVP9-BUG-FLU-00030 — Timing qualifiers cannot enforce the idle-host gate on Windows

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** benchmark
- **Raised:** 2026-08-22T18:08:12Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260822T180840Z-795f80a2
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00030-run-fix-20260822T180840Z-795f80a2
- **Owner base:** b7142cf499949b3c7bd5f9ebcad0ff644cee02c5
- **Owner fingerprint:** -
- **Owner since:** 2026-08-22T18:08:40Z
- **Owner until:** 2026-08-22T20:08:40Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T18:08:12Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

On an otherwise idle Windows x86_64 host, qualify-spatial-svc.sh and qualify-production-encoder.sh cannot read /proc/loadavg or vm.loadavg, while encode-headtohead has the same Unix-only current_load1 implementation. Every timing phase therefore fails closed before measuring. Expected: the checked-in acceptance qualifier should measure a conservative Windows host-idle equivalent and preserve the 0.75 busy-core ceiling.

## Fix

<unfixed — raised only>

## Notes
