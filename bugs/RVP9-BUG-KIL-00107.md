# RVP9-BUG-KIL-00107 — BD-rate singular fits silently produce a finite zero result

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** bench/bdrate
- **Raised:** 2026-08-22T19:30:33Z
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
- **State history:** Open (2026-08-22T19:30:33Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Four or more RD points can still contain fewer than four distinct PSNR values. crates/rvp9-bench/src/bin/bdrate.rs:434-477 returns zero coefficients when the normal-equation matrix is singular, and bd_rate at lines 371-412 integrates those coefficients without detecting failure. measured_bd_rate then accepts the finite 0.0 result at lines 415-424. Expected: rank-deficient curves fail the measurement. Actual: an undefined curve can be reported and written as 0 percent BD-rate. Make the fit fallible or use the shared harness solver, propagate the error, and add a repeated-PSNR regression. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
