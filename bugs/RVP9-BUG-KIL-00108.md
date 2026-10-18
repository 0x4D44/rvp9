# RVP9-BUG-KIL-00108 — Throughput CSV accepts corrupt and non-finite baselines

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** bench/throughput ratchet
- **Raised:** 2026-08-22T19:30:41Z
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
- **State history:** Open (2026-08-22T19:30:41Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T00:53:09Z, deltic:auto role=fix run=fix-20261016T004105Z-6f8a5d31 branch=task/bug-RVP9-BUG-KIL-00108-run-fix-20261016T004105Z-6f8a5d31 code=22a2b2db5b229203f6a198366559e82226ab0048 gate=manual) -> Closed (2026-10-18T16:02:50Z, deltic:manual role=verify run=verify-20261018T155745Z-5619443c verified_fix_run=fix-20261016T004105Z-6f8a5d31 verdict=close)

## Observation

crates/rvp9-bench/src/throughput.rs:454-470 writes caller-controlled host, profile, and fixture fields without CSV escaping. read_csv at lines 501-553 splits on commas, accepts extra columns, and accepts NaN or infinity as f64 metrics. check_metric_ratchet at lines 608-624 treats NaN comparisons as within budget. Expected: only structurally valid rows with finite positive metrics can become baselines. Actual: a comma-bearing fixture name can shift fields, and a non-finite metric can make the regression gate pass. Use a proper fixed-schema CSV codec, require exactly the expected columns, validate finite positive metrics, and add comma, NaN, and infinity regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T16:02:50Z, deltic:auto run=verify-20261018T155745Z-5619443c verified_fix_run=fix-20261016T004105Z-6f8a5d31 verdict=close)

Verifier note: Manual verify: Independent session01a14fa5, not fixer; base283ea929 includes22a2b2db. Original CSV/nonfinite observation:9 regressions pass. Six root reversals each execute1 and assertion-fail: raw writer quoting, physical-line split reader, extra columns, reader metric validation, append prevalidation, unchecked ratchet operands/product. Observed rows1vs2,2vs1,16vs1; literal mismatch; missing InvalidInput; invalid operands Ok. Exact bytes restored after each, scoped diff empty. Final throughput20 pass. Shared gates and limitations plus raw assertions in2026.10.18 verification journal.

## Notes
