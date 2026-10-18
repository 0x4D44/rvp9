# RVP9-BUG-KIL-00107 — BD-rate singular fits silently produce a finite zero result

- **State:** Closed
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
- **State history:** Open (2026-08-22T19:30:33Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T04:43:08Z, deltic:auto role=fix run=fix-20261016T043838Z-47fe7116 branch=task/bug-RVP9-BUG-KIL-00107-run-fix-20261016T043838Z-47fe7116 code=bfe2cf7a3de234ad21a6c48a9a55c3aaf9ca6a56 gate=manual) -> Closed (2026-10-18T16:01:16Z, deltic:manual role=verify run=verify-20261018T155648Z-a6df36e4 verified_fix_run=fix-20261016T043838Z-47fe7116 verdict=close)

## Observation

Four or more RD points can still contain fewer than four distinct PSNR values. crates/rvp9-bench/src/bin/bdrate.rs:434-477 returns zero coefficients when the normal-equation matrix is singular, and bd_rate at lines 371-412 integrates those coefficients without detecting failure. measured_bd_rate then accepts the finite 0.0 result at lines 415-424. Expected: rank-deficient curves fail the measurement. Actual: an undefined curve can be reported and written as 0 percent BD-rate. Make the fit fallible or use the shared harness solver, propagate the error, and add a repeated-PSNR regression. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T16:01:16Z, deltic:auto run=verify-20261018T155648Z-a6df36e4 verified_fix_run=fix-20261016T043838Z-47fe7116 verdict=close)

Verifier note: Manual verify: Independent session01a14fa5, not fixer; base283ea929 includesbfe2cf7a. Original rank-deficient >=4 point curves: six measured_bdrate tests pass. Restore parent solver behavior (Some adaptation only for current signature): four singular-anchor/test/both/realistic-PSNR assertions fail as undefined curves return BD-rate; full-rank and invalid-input controls pass. Exact source restored/diff empty; restored6 pass, lead executable rerun6 pass. Lead checked distinct-PSNR rank rejection and Option propagation from both fits. Shared gates/limits and raw proof in2026.10.18 verification jo...

## Notes
