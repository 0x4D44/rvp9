# RVP9-BUG-FLUX-00004 — Bit-accounting unit tests race on the process-wide enable switch

- **State:** Closed
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) → Fixed (2026-07-30, `ecb6d88`, fixed by OpenAI Codex) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

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

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Root cause addressed at the right layer: the process-wide `BITACCT_ON` switch is now taken
under a mutex (`crates/rvp9-core/src/entropy/bitacct.rs:154`), and `BITACCT_ON` is mutated in
exactly the three tests that hold it -- verified by grepping the whole `crates/` tree, so the
lock covers every writer. Drop order is correct: `_guard` is declared before the `Restore`
struct, so the switch is restored before the lock releases.
`entropy::bitacct::tests::full_coverage_sweep`, `::bool_writer_boundaries_reset_to_other` and
`::suppression_skips_recording_and_restores_enabled_state` all pass.
Residual noted, not blocking: `.lock().unwrap()` means one genuine failure poisons the mutex
and cascades into the other two as unwrap panics, obscuring the real failure in a log.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
