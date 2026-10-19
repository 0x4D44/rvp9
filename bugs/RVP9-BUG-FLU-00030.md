# RVP9-BUG-FLU-00030 — Timing qualifiers cannot enforce the idle-host gate on Windows

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** benchmark
- **Raised:** 2026-08-22T18:08:12Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T070514Z-71393735
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00030-run-verify-20261019T070514Z-71393735
- **Owner base:** 0605dcfac478b2f7b8f17266efba092432aa0fc7
- **Owner fingerprint:** sha256:10ce542884ccadd612706157d0bd30e24b30c1a7652b816b2c7d3b28d6213cc3
- **Owner since:** 2026-10-19T07:05:14Z
- **Owner until:** 2026-10-19T11:05:14Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T18:08:12Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-22T18:20:49Z, deltic:auto role=fix run=fix-20260822T180840Z-795f80a2 branch=task/bug-RVP9-BUG-FLU-00030-run-fix-20260822T180840Z-795f80a2 code=36eec127 gate=manual)

## Observation

On an otherwise idle Windows x86_64 host, qualify-spatial-svc.sh and qualify-production-encoder.sh cannot read /proc/loadavg or vm.loadavg, while encode-headtohead has the same Unix-only current_load1 implementation. Every timing phase therefore fails closed before measuring. Expected: the checked-in acceptance qualifier should measure a conservative Windows host-idle equivalent and preserve the 0.75 busy-core ceiling.

## Fix

<unfixed — raised only>

## Notes
