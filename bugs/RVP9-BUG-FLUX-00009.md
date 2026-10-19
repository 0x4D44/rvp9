# RVP9-BUG-FLUX-00009 — f02 oracle sink test races global environment state

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tests/encoder f02 oracle
- **Raised:** 2026-08-07T14:03:38Z
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
- **State history:** Open (2026-08-07T14:03:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-07T16:50:12Z, deltic:auto role=fix run=fix-20260807T164647Z-p56265-n874415000-c1 branch=task/bug-RVP9-BUG-FLUX-00009-run-fix-20260807T164647Z-p56265-n874415000-c1 code=51234e3a gate=manual) -> Closed (2026-10-19T06:11:44Z, deltic:manual role=verify run=verify-20261019T060225Z-ab27e285 verified_fix_run=fix-20260807T164647Z-p56265-n874415000-c1 verdict=close)

## Observation

Observation: 'cargo test -p rvp9-encoder --locked' failed once at crates/rvp9-encoder/src/f02_oracle.rs:588 in sink_on_records_and_drains: the drained record count was 0 instead of 2. The exact isolated test then passed, and a full encoder-suite rerun passed. Inference: tests that share the process-global RVP9_F02_ORACLE environment variable can race and disable the sink during this test. Expected: the test remains deterministic under the suite's normal parallel execution.

## Fix

Commit `51234e3a` makes all three f02 unit tests that mutate the process-global
enable state hold the encoder crate's shared environment-test mutex. This
serializes them with encoder tests that call `f02_oracle::init_from_env`, so a
parallel test can no longer disable the sink between record calls.

Focused f02 tests, the full encoder library, workspace formatting, workspace
all-target clippy, workspace tests, and doctests passed. The bug remains
`Fixed` until a different verifier independently confirms and closes it.

### Verification summary (2026-10-19T06:11:44Z, deltic:auto run=verify-20261019T060225Z-ab27e285 verified_fix_run=fix-20260807T164647Z-p56265-n874415000-c1 verdict=close)

Verifier note: Manual verify: Independent manual verification by Codex. Recorded code SHA 51234e3a is absent from origin/main; matching fix commit 6a1e694fca8e57fa58942181abf4a080df6de794 is an ancestor. Original cargo test -p rvp9-encoder --locked passed (unit 824 passed, 1 ignored; integration targets and doctests passed); focused f02_oracle::tests::sink_on_records_and_drains selected 1 and passed. Fails-before: removed the three env_test_mutex guards added by 6a1e694f; the full encoder run selected 825 tests, 823 passed, 1 ignored, and capture_isolates_caller_jobs_and_reused_worker_tls failed its F02 reco...

## Notes
