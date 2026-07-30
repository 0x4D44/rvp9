# RVP9-BUG-FLUX-00004 — Bit-accounting unit tests race on the process-wide enable switch

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/entropy tests
- **Raised:** 2026-07-30
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
- **Attempts:** fix=1, doubt=0, indeterminate=0
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) → Fixed (2026-07-30, `ecb6d88`, fixed by OpenAI Codex)

## Observation

Observed on aarch64-apple-darwin with Rust 1.95.0:
`cargo test -p rvp9-core --lib entropy::bitacct::tests:: --locked` passes
`full_coverage_sweep` in isolation but fails it when the module's three tests
run in parallel. The tests independently store and restore the process-wide
`BITACCT_ON` switch, so another test can disable accounting between
`set_category` and `record`; the observed failure was `expected 9, got 0`.
Expected: the documented workspace gate is deterministic. Actual: normal test
parallelism races the bit-accounting test controls.

## Fix

Commit `ecb6d88` adds one test-only mutex shared by all three tests that mutate
`BITACCT_ON`. The previously failing three-test selection passed 20 consecutive
parallel runs, and `cargo test --workspace --locked` passes.

## Notes
