# RVP9-BUG-KILN-00055 - CLI integration tests wait unbounded on child processes

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/tests
- **Raised:** 2026-07-07
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T195724Z-p52206-n504826000-c1 branch=task/bug-RVP9-BUG-KILN-00055-run-fix-20260730T195724Z-p52206-n504826000-c1 code=72d8ef7ad52b671aee849c0b895aca7e7fa55e3e gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The CLI integration tests invoke child binaries with unbounded `Command::output()` waits and no stdin closure helper, timeout, or kill-on-timeout path.

Examples include `crates/rvp9-cli/tests/cli_arg_coverage.rs:1056`-`1060`, `crates/rvp9-cli/tests/cli_arg_coverage.rs:1927`-`1931`, and `crates/rvp9-cli/tests/cli_roundtrip.rs:277`. A focused search found many `.output()` calls across `crates/rvp9-cli/tests` and no timeout or `Stdio::null()` helper in that directory.

Expected: subprocess-based tests should bound runtime, close stdin, kill hung children, and cap captured output. Actual: a hung CLI child can hang the test process indefinitely.

## Fix

Centralize CLI test execution behind a helper that sets stdin to null, enforces a timeout, kills on timeout, and returns captured output. Port all direct `Command::output()` calls through it.

## Notes

`RVP9-BUG-KILN-00037` tracks analogous unbounded external waits in harness diagnostic binaries. This bug tracks the CLI crate's integration-test harness.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Every child spawn in the CLI test directory now goes through `CommandExt::bounded_output`/
`bounded_status`, which nulls stdin, drains both pipes on capped threads, polls to a deadline and
then kills and waits (`crates/rvp9-cli/tests/common/mod.rs:25`, `:48`). A grep of
`crates/rvp9-cli/tests` for raw `.output()`/`.status()`/`.spawn()` found ZERO outside the helper
-- 97 bounded call sites in `cli_arg_coverage.rs` and 13 in `cli_roundtrip.rs`, including all
three sites the observation named. Capped drain threads mean an oversized child cannot deadlock
the pipe.
`bounded_output_kills_a_timed_out_child`, `bounded_output_caps_but_drains_each_stream` and
`bounded_status_uses_the_same_runner` all observed passing.
Two residuals noted, neither blocking: the timeout test asserts only the error kind and elapsed
time, never that the child actually died (kill/wait results are discarded with `let _ =`); and
the fixed 30s default could turn a legitimately slow encode on a loaded box into a spurious
`TimedOut`.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
