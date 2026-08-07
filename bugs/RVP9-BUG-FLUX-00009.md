# RVP9-BUG-FLUX-00009 — f02 oracle sink test races global environment state

- **State:** Open
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
- **State history:** Open (2026-08-07T14:03:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: 'cargo test -p rvp9-encoder --locked' failed once at crates/rvp9-encoder/src/f02_oracle.rs:588 in sink_on_records_and_drains: the drained record count was 0 instead of 2. The exact isolated test then passed, and a full encoder-suite rerun passed. Inference: tests that share the process-global RVP9_F02_ORACLE environment variable can race and disable the sink during this test. Expected: the test remains deterministic under the suite's normal parallel execution.

## Fix

<unfixed — raised only>

## Notes
