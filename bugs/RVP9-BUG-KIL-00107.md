# RVP9-BUG-KIL-00107 — BD-rate singular fits silently produce a finite zero result

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** bench/bdrate
- **Raised:** 2026-08-22T19:30:33Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T155648Z-a6df36e4
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00107-run-verify-20261018T155648Z-a6df36e4
- **Owner base:** 7d71e9de3682da34c966c08cd819945103d124a8
- **Owner fingerprint:** sha256:dbd8d3b878f8073448c290de3d0b8a96dfd6ce4f835d025e4ea7fd9b907b27f0
- **Owner since:** 2026-10-18T15:56:48Z
- **Owner until:** 2026-10-18T19:56:48Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:30:33Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T04:43:08Z, deltic:auto role=fix run=fix-20261016T043838Z-47fe7116 branch=task/bug-RVP9-BUG-KIL-00107-run-fix-20261016T043838Z-47fe7116 code=bfe2cf7a3de234ad21a6c48a9a55c3aaf9ca6a56 gate=manual)

## Observation

Four or more RD points can still contain fewer than four distinct PSNR values. crates/rvp9-bench/src/bin/bdrate.rs:434-477 returns zero coefficients when the normal-equation matrix is singular, and bd_rate at lines 371-412 integrates those coefficients without detecting failure. measured_bd_rate then accepts the finite 0.0 result at lines 415-424. Expected: rank-deficient curves fail the measurement. Actual: an undefined curve can be reported and written as 0 percent BD-rate. Make the fit fallible or use the shared harness solver, propagate the error, and add a repeated-PSNR regression. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
