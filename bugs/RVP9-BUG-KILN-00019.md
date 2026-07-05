# RVP9-BUG-KILN-00019 - Non-ignored encoder tests have false-green oracles

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/test-oracles
- **Raised:** 2026-07-05
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`)

## Observation

Several non-ignored encoder tests can pass without proving the property named by the test:

- `w_lag_in_frames_nonzero_rejects_sync_encode_frame` at `crates/rvp9-encoder/src/frame_loop.rs:9556` stores an error from `expect_err`, then computes `matches!(err, EncodeError::InvalidConfig(_))` at `crates/rvp9-encoder/src/frame_loop.rs:9585` and discards the boolean. The test fails if `encode_frame` succeeds, but passes for any error variant.
- `phase2_newmv_reuse_fires` at `crates/rvp9-encoder/tests/phase2_reuse_positive_control.rs:55` returns successfully when `foreman_cif` is absent at line 57.
- `a21_ml_partition_breakout_fires_and_decodes` at `crates/rvp9-encoder/tests/w_ml_partition.rs:224` returns successfully when `testsrc_720p30.y4m` is missing or unreadable at lines 226 and 240.

These tests are not ignored, so a normal local test run can report green while the intended oracle did not run or did not assert the expected condition.

## Fix

Open. Wrap the discarded `matches!` call in an assertion. For the corpus-dependent tests, use deterministic synthetic fixtures, make missing required corpus a hard failure, or mark the tests ignored/informational so they do not count as green proof in normal runs.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.
