# RVP9-BUG-FLU-00047 — Production timing matrix exceeds Deltic quiet leases

- **State:** Closed
- **Priority:** Must
- **Severity:** Medium
- **Area:** benchmark
- **Raised:** 2026-10-17T13:58:37Z
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
- **State history:** Open (2026-10-17T13:58:37Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T18:28:43Z, deltic:auto role=fix run=fix-20261018T174427Z-66908827 branch=task/bug-RVP9-BUG-FLU-00047-run-fix-20261018T174427Z-66908827 code=056ddb1839c3caa22bffcc048f06305e4cfb3974 gate=manual) -> Closed (2026-10-18T21:50:27Z, deltic:manual role=verify run=verify-20261018T214348Z-ae94c12c verified_fix_run=fix-20261018T174427Z-66908827 verdict=close)

## Observation

On the exact fb5c0c917038db6c640b3dbc7c2b84b567233333 candidate on native Windows NMI, the production speed phase requires 75 groups and the recorded host-load guard samples for 60 seconds per group, a 75-minute minimum. Deltic quiet leases are capped at 20 minutes, and the timing phase refuses resume after its first measurement to avoid mixing host states. Expected: the documented complete timing gate can run under the documented quiet policy without combining measurements from different host states. Actual: the complete phase cannot fit in one lease and a partial phase cannot resume; the current record therefore started no timing campaign and has no performance result. See recovered-validation-20261017/postfix-main-qualification/windows-nmi-current-source-pgo-qualification.json (codec_timing) and the production runner timing-resume contract.

## Fix

The production runner now resumes speed at complete speed-group boundaries and latency at complete feature/mode/target/CPU groups. Receipts bind each completed batch to the sealed run fingerprint and a stable hash of its retained artifacts. Resume validates receipts and latency row/load evidence, reuses only complete batches, and preserves interrupted batches for whole-group retry in a new attempt. The phase aggregate is rebuilt from sealed batches and published with its canonical summary files in one directory rename.

The regression assertion failed before the change with `resume self-test: sealed timing batch could not resume` (exit 1). After the change, `--self-test-resume` and the stubbed `--self-test-timing` passed; the latter verified reuse of completed speed and latency batches, whole-group retries, preserved partial attempts, and duplicate-free summaries. No real encoder timing or performance result was produced.

### Verification summary (2026-10-18T21:50:27Z, deltic:auto run=verify-20261018T214348Z-ae94c12c verified_fix_run=fix-20261018T174427Z-66908827 verdict=close)

Verifier note: Manual verify: Independent verification on 5b4b3f62: reviewed sealed batch receipts, per-group load evidence, fingerprint binding and complete-group retry. bash scripts/qualify-production-encoder.sh --self-test-resume and --self-test-timing pass; timing checks reuse one sealed speed/latency group, retry partial groups, preserve failed attempts, and publish each row once. Reversing only timing_resume_state to pre-fix rejection selected and failed sealed-timing-batch-could-not-resume assertion; restored source and resume test passed. bash -n passed. This verifies resumability contract, not real ...

## Notes
