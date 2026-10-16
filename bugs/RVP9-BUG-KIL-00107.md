# RVP9-BUG-KIL-00107 — BD-rate singular fits silently produce a finite zero result

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** bench/bdrate
- **Raised:** 2026-08-22T19:30:33Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T043838Z-47fe7116
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00107-run-fix-20261016T043838Z-47fe7116
- **Owner base:** b1c309fe0a494d9171bb068c9ba9e078401dca4d
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T04:38:38Z
- **Owner until:** 2026-10-16T08:38:38Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:30:33Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Four or more RD points can still contain fewer than four distinct PSNR values. crates/rvp9-bench/src/bin/bdrate.rs:434-477 returns zero coefficients when the normal-equation matrix is singular, and bd_rate at lines 371-412 integrates those coefficients without detecting failure. measured_bd_rate then accepts the finite 0.0 result at lines 415-424. Expected: rank-deficient curves fail the measurement. Actual: an undefined curve can be reported and written as 0 percent BD-rate. Make the fit fallible or use the shared harness solver, propagate the error, and add a repeated-PSNR regression. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
