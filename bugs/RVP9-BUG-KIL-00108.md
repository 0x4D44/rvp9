# RVP9-BUG-KIL-00108 — Throughput CSV accepts corrupt and non-finite baselines

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** bench/throughput ratchet
- **Raised:** 2026-08-22T19:30:41Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T004105Z-6f8a5d31
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00108-run-fix-20261016T004105Z-6f8a5d31
- **Owner base:** ec90c75054ede4a5adcf8b40fedcb4b63202b5a6
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T00:41:05Z
- **Owner until:** 2026-10-16T04:41:05Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:30:41Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

crates/rvp9-bench/src/throughput.rs:454-470 writes caller-controlled host, profile, and fixture fields without CSV escaping. read_csv at lines 501-553 splits on commas, accepts extra columns, and accepts NaN or infinity as f64 metrics. check_metric_ratchet at lines 608-624 treats NaN comparisons as within budget. Expected: only structurally valid rows with finite positive metrics can become baselines. Actual: a comma-bearing fixture name can shift fields, and a non-finite metric can make the regression gate pass. Use a proper fixed-schema CSV codec, require exactly the expected columns, validate finite positive metrics, and add comma, NaN, and infinity regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
