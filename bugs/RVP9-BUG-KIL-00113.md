# RVP9-BUG-KIL-00113 — Realtime accepts an infinite budget that makes every finite run pass

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/realtime validation
- **Raised:** 2026-08-22T19:31:33Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T161832Z-7a674800
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00113-run-verify-20261018T161832Z-7a674800
- **Owner base:** 4cb660133c50815eb680f8ca69064a0c3dc4d1d5
- **Owner fingerprint:** sha256:0f4739626e0a054babd5a33370775f98f3b95994e30abd7a6c0638307753b666
- **Owner since:** 2026-10-18T16:18:32Z
- **Owner until:** 2026-10-18T20:18:32Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:31:33Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T04:04:10Z, deltic:auto role=fix run=fix-20261016T035752Z-af34a473 branch=task/bug-RVP9-BUG-KIL-00113-run-fix-20261016T035752Z-af34a473 code=26ac4ff115a1b042bfaab212629a0d013ca5dd29 gate=manual)

## Observation

crates/rvp9-bench/src/bin/realtime.rs:115-117 parses budget-ms as f64 without requiring it to be finite and positive. The pass check at lines 425-431 accepts every finite average when the user supplies infinity, while p99-budget-ms already enforces the correct finite-positive contract at lines 184-190. Expected: invalid budgets fail argument validation. Actual: an infinite primary budget turns the performance gate into an unconditional pass. Apply the same finite-positive validation to budget-ms and add NaN, positive infinity, negative infinity, zero, and negative-value parser regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
