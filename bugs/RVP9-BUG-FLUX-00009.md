# RVP9-BUG-FLUX-00009 — f02 oracle sink test races global environment state

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tests/encoder f02 oracle
- **Raised:** 2026-08-07T14:03:38Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T060225Z-ab27e285
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00009-run-verify-20261019T060225Z-ab27e285
- **Owner base:** 426057a0168265bb8389734c1e4100bb7f53bd39
- **Owner fingerprint:** sha256:045e899dd491587864faa794610a259c004a9265e11c87236b2a28a589109796
- **Owner since:** 2026-10-19T06:02:25Z
- **Owner until:** 2026-10-19T10:02:25Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-07T14:03:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-07T16:50:12Z, deltic:auto role=fix run=fix-20260807T164647Z-p56265-n874415000-c1 branch=task/bug-RVP9-BUG-FLUX-00009-run-fix-20260807T164647Z-p56265-n874415000-c1 code=51234e3a gate=manual)

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

## Notes
