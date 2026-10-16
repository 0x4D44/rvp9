# RVP9-BUG-KIL-00113 — Realtime accepts an infinite budget that makes every finite run pass

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/realtime validation
- **Raised:** 2026-08-22T19:31:33Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T035752Z-af34a473
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00113-run-fix-20261016T035752Z-af34a473
- **Owner base:** 2233c4ff1ad4febc3583b577d80614704915c9c6
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T03:57:52Z
- **Owner until:** 2026-10-16T07:57:52Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:31:33Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

crates/rvp9-bench/src/bin/realtime.rs:115-117 parses budget-ms as f64 without requiring it to be finite and positive. The pass check at lines 425-431 accepts every finite average when the user supplies infinity, while p99-budget-ms already enforces the correct finite-positive contract at lines 184-190. Expected: invalid budgets fail argument validation. Actual: an infinite primary budget turns the performance gate into an unconditional pass. Apply the same finite-positive validation to budget-ms and add NaN, positive infinity, negative infinity, zero, and negative-value parser regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
