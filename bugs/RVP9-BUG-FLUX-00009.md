# RVP9-BUG-FLUX-00009 — f02 oracle sink test races global environment state

- **State:** Fixed
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
- **State history:** Open (2026-08-07T14:03:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-07T16:50:12Z, deltic:auto role=fix run=fix-20260807T164647Z-p56265-n874415000-c1 branch=task/bug-RVP9-BUG-FLUX-00009-run-fix-20260807T164647Z-p56265-n874415000-c1 code=51234e3a gate=manual)

## Observation

Observation: 'cargo test -p rvp9-encoder --locked' failed once at crates/rvp9-encoder/src/f02_oracle.rs:588 in sink_on_records_and_drains: the drained record count was 0 instead of 2. The exact isolated test then passed, and a full encoder-suite rerun passed. Inference: tests that share the process-global RVP9_F02_ORACLE environment variable can race and disable the sink during this test. Expected: the test remains deterministic under the suite's normal parallel execution.

## Fix

<unfixed — raised only>

## Notes
