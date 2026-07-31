# RVP9-BUG-KILN-00057 - decode-throughput exits successfully when measurement or CSV recording fails

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/decode-throughput
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T184605Z-p81009-n041713000-c1 branch=task/bug-RVP9-BUG-KILN-00057-run-fix-20260730T184605Z-p81009-n041713000-c1 code=013ddfd44077f8d2f39a94cd744771a906d1f36c gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5); residual split to RVP9-BUG-KILN-00066)

## Observation

`decode-throughput` returns exit 0 after failed measurement states that should make the command unusable as a recorder.

If no fixture produces timed samples, `crates/rvp9-bench/src/bin/decode-throughput.rs:152`-`155` prints "nothing to record" and returns from `main`. If `throughput::append_csv` fails, `crates/rvp9-bench/src/bin/decode-throughput.rs:163`-`172` prints the write failure and also returns. A partial run with at least one failed fixture only prints a warning at `crates/rvp9-bench/src/bin/decode-throughput.rs:176`-`178`.

Expected: a benchmark command should exit non-zero when it produced no measurement rows or could not write the requested CSV. Actual: callers can treat a missing or unrecorded measurement as success.

## Fix

Exit non-zero when `results.is_empty()` or CSV append fails. Consider also returning non-zero for fixture errors when the user passed an explicit `--fixture`; the default multi-fixture sweep can still choose whether partial data is acceptable.

## Notes

The ratchet test may remain the performance gate, but the measurement binary should still distinguish "measurement recorded" from "no data / no CSV".

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Both named conditions now return `ExitCode::FAILURE` directly from `main`
(`crates/rvp9-bench/src/bin/decode-throughput.rs:153`, `:171`; signature changed to
`fn main() -> ExitCode`) -- no `let _ =`, no `.ok()`, nothing between the return and process
exit. The whole `main` body was read to confirm no intermediate swallowing.
`missing_explicit_fixture_exits_nonzero` and `csv_append_failure_exits_nonzero` both observed
passing; they run the real `CARGO_BIN_EXE_decode-throughput` and assert both a non-zero status
and the specific stderr string, so they would fail against the old `return;` code.
Residual split to RVP9-BUG-KILN-00066: the third symptom this record named
(`decode-throughput.rs:176-178`, a partial run with a failed fixture) is UNCHANGED -- it still
prints a note and returns `ExitCode::SUCCESS`. This record's Fix section flagged that limb as
optional, and under an explicit `--fixture` the failure now collapses to `results.is_empty()`
and is fatal, so the explicit-fixture case is covered.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
