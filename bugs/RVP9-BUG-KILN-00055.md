# RVP9-BUG-KILN-00055 - CLI integration tests wait unbounded on child processes

- **State:** Open
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`)

## Observation

The CLI integration tests invoke child binaries with unbounded `Command::output()` waits and no stdin closure helper, timeout, or kill-on-timeout path.

Examples include `crates/rvp9-cli/tests/cli_arg_coverage.rs:1056`-`1060`, `crates/rvp9-cli/tests/cli_arg_coverage.rs:1927`-`1931`, and `crates/rvp9-cli/tests/cli_roundtrip.rs:277`. A focused search found many `.output()` calls across `crates/rvp9-cli/tests` and no timeout or `Stdio::null()` helper in that directory.

Expected: subprocess-based tests should bound runtime, close stdin, kill hung children, and cap captured output. Actual: a hung CLI child can hang the test process indefinitely.

## Fix

Centralize CLI test execution behind a helper that sets stdin to null, enforces a timeout, kills on timeout, and returns captured output. Port all direct `Command::output()` calls through it.

## Notes

`RVP9-BUG-KILN-00037` tracks analogous unbounded external waits in harness diagnostic binaries. This bug tracks the CLI crate's integration-test harness.
