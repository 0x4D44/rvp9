# RVP9-BUG-KIL-00087 — PSNR delta summary returns non-finite statistics

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** harness/bdrate
- **Raised:** 2026-08-22T13:20:52Z
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
- **State history:** Open (2026-08-22T13:20:52Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T07:59:11Z, deltic:auto role=fix run=fix-20261016T074641Z-b354a429 branch=task/bug-RVP9-BUG-KIL-00087-run-fix-20261016T074641Z-b354a429 code=e61bd7e2be6e2971edf589a932678589ae31ac93 gate=manual) -> Closed (2026-10-18T17:03:04Z, deltic:manual role=verify run=verify-20261018T165858Z-f3876347 verified_fix_run=fix-20261016T074641Z-b354a429 verdict=close)

## Observation

`psnr_at_bitrate` rejects non-finite samples in the reference curve at
`crates/rvp9-harness/src/bdrate.rs:131-139`, but `psnr_delta_summary` does not
validate the rvp9 sample's `psnr_r` before subtracting it at lines 221-224. A
covered rvp9 point containing `NaN` or infinity is pushed into `deltas`, and
the function returns `Some(PsnrDeltaSummary)` with non-finite mean/min/max at
lines 227-241.

Expected: invalid samples produce no quality signal. Actual: the API returns a
present summary whose statistics cannot be meaningfully compared or reported.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T17:03:04Z, deltic:auto run=verify-20261018T165858Z-f3876347 verified_fix_run=fix-20261016T074641Z-b354a429 verdict=close)

Verifier note: Manual verify: Independent verification on the fixed build: 12 focused interpolation/summary tests passed. Reverting only psnr_delta_summary to its pre-fix implementation left 5 controls passing and failed 3 retained regressions, which observed a NaN partial summary, infinite statistics from finite-input overflow, and an invalid-bitrate partial result. Exact source restored; scoped diff clean. Adjacent cubic BD-rate fitting remains outside this record.

## Notes

Validate the complete rvp9 curve before interpolation and return `None` for any
non-positive/non-finite bitrate or non-finite PSNR, matching the reference-curve
policy. Add `NaN` and infinity regressions. Confirmed statically; no tests ran.
