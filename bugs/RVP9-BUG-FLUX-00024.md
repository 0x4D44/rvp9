# RVP9-BUG-FLUX-00024 — Encoder diagnostic environment variables race parallel tests

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/test isolation
- **Raised:** 2026-08-13T03:42:41Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261015T231847Z-e15a9711
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00024-run-fix-20261015T231847Z-e15a9711
- **Owner base:** 57199e3d88496eba1169e36285cff19c85eec6bb
- **Owner fingerprint:** -
- **Owner since:** 2026-10-15T23:18:47Z
- **Owner until:** 2026-10-16T03:18:47Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T03:42:41Z, raised via `deltic bugs new`)

## Observation

`cargo test --release -p rvp9-encoder --lib` intermittently failed
`frame_loop::tests::w_lag_zero_tight_and_strided_paths_match` at frame zero:
the tight and strided encoders produced different keyframe packets from the
same visible pixels. The immediately preceding full-suite run and the next run
both passed.

The failing test encodes its tight frame and then its strided frame serially.
Other parallel tests under `crates/rvp9-encoder/src/frame_loop.rs` mutate
process-global diagnostics such as `RVP9_EXACT_KF_SUB8` between those calls.
`env_test_mutex` serialises only tests that opt into the mutex; ordinary encode
tests do not hold it while production policy reads those variables. Therefore
one encoder can observe a diagnostic override and the other can miss it.

Expected: parallel unit tests cannot change another test's encoder decisions.
Actual: process-global diagnostic overrides can make otherwise identical
encodes differ and fail the full regression gate nondeterministically.

## Fix

<unfixed — raised only>

## Notes

Discovered while adding a spatial-SVC policy test. That new test was changed
to exercise a pure predicate and no longer mutates the environment. The
underlying older setters and per-encode readers remain and need an isolated
test-only override mechanism or construction-time capture.
