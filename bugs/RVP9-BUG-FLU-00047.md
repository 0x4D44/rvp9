# RVP9-BUG-FLU-00047 — Production timing matrix exceeds Deltic quiet leases

- **State:** Fixed
- **Priority:** Must
- **Severity:** Medium
- **Area:** benchmark
- **Raised:** 2026-10-17T13:58:37Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T214348Z-ae94c12c
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00047-run-verify-20261018T214348Z-ae94c12c
- **Owner base:** f24e2d108f05b11b9ff99107d028802ac5f9e1e1
- **Owner fingerprint:** sha256:128cba1aada9b56a6d1e577988d84f76733ebe6cefb390db906efb631c198376
- **Owner since:** 2026-10-18T21:43:48Z
- **Owner until:** 2026-10-19T01:43:48Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-17T13:58:37Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T18:28:43Z, deltic:auto role=fix run=fix-20261018T174427Z-66908827 branch=task/bug-RVP9-BUG-FLU-00047-run-fix-20261018T174427Z-66908827 code=056ddb1839c3caa22bffcc048f06305e4cfb3974 gate=manual)

## Observation

On the exact fb5c0c917038db6c640b3dbc7c2b84b567233333 candidate on native Windows NMI, the production speed phase requires 75 groups and the recorded host-load guard samples for 60 seconds per group, a 75-minute minimum. Deltic quiet leases are capped at 20 minutes, and the timing phase refuses resume after its first measurement to avoid mixing host states. Expected: the documented complete timing gate can run under the documented quiet policy without combining measurements from different host states. Actual: the complete phase cannot fit in one lease and a partial phase cannot resume; the current record therefore started no timing campaign and has no performance result. See recovered-validation-20261017/postfix-main-qualification/windows-nmi-current-source-pgo-qualification.json (codec_timing) and the production runner timing-resume contract.

## Fix

The production runner now resumes speed at complete speed-group boundaries and latency at complete feature/mode/target/CPU groups. Receipts bind each completed batch to the sealed run fingerprint and a stable hash of its retained artifacts. Resume validates receipts and latency row/load evidence, reuses only complete batches, and preserves interrupted batches for whole-group retry in a new attempt. The phase aggregate is rebuilt from sealed batches and published with its canonical summary files in one directory rename.

The regression assertion failed before the change with `resume self-test: sealed timing batch could not resume` (exit 1). After the change, `--self-test-resume` and the stubbed `--self-test-timing` passed; the latter verified reuse of completed speed and latency batches, whole-group retries, preserved partial attempts, and duplicate-free summaries. No real encoder timing or performance result was produced.

## Notes
