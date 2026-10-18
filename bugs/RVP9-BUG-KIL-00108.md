# RVP9-BUG-KIL-00108 — Throughput CSV accepts corrupt and non-finite baselines

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** bench/throughput ratchet
- **Raised:** 2026-08-22T19:30:41Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T155745Z-5619443c
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00108-run-verify-20261018T155745Z-5619443c
- **Owner base:** a8a64222db7d0687c3c3e37012b490cd8d7b49ec
- **Owner fingerprint:** sha256:79fe906ea47c6eb70f7cce82c644b2e066202c2fbf3524a9083b4d1f0d44e8ac
- **Owner since:** 2026-10-18T15:57:45Z
- **Owner until:** 2026-10-18T19:57:45Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:30:41Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T00:53:09Z, deltic:auto role=fix run=fix-20261016T004105Z-6f8a5d31 branch=task/bug-RVP9-BUG-KIL-00108-run-fix-20261016T004105Z-6f8a5d31 code=22a2b2db5b229203f6a198366559e82226ab0048 gate=manual)

## Observation

crates/rvp9-bench/src/throughput.rs:454-470 writes caller-controlled host, profile, and fixture fields without CSV escaping. read_csv at lines 501-553 splits on commas, accepts extra columns, and accepts NaN or infinity as f64 metrics. check_metric_ratchet at lines 608-624 treats NaN comparisons as within budget. Expected: only structurally valid rows with finite positive metrics can become baselines. Actual: a comma-bearing fixture name can shift fields, and a non-finite metric can make the regression gate pass. Use a proper fixed-schema CSV codec, require exactly the expected columns, validate finite positive metrics, and add comma, NaN, and infinity regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
