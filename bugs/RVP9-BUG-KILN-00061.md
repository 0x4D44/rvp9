# RVP9-BUG-KILN-00061 - soak cross-decode uses shared temp files and an unbounded vpxdec wait

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/soak
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T190353Z-p998-n865634000-c1 branch=task/bug-RVP9-BUG-KILN-00061-run-fix-20260730T190353Z-p998-n865634000-c1 code=30e5dabe75164a051ebf6f35dae72b73ff2ad86c gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The soak harness cross-decode leg writes deterministic filenames in the shared system temp directory and waits for `vpxdec` without a timeout.

`crates/rvp9-bench/src/bin/soak.rs:388`-`397` builds `rvp9_soak_iter{iter_idx}.ivf` and `.yuv` under `std::env::temp_dir()`. Two concurrent soak runs with the same iteration index can truncate, overwrite, read, or remove each other's files. `crates/rvp9-bench/src/bin/soak.rs:397`-`402` then runs `Command::new(vpxdec).output()` with no timeout or kill path; cleanup happens only after that call returns at `crates/rvp9-bench/src/bin/soak.rs:403`-`404`.

Expected: soak's optional external decoder leg should use isolated per-run temp paths and bounded child-process execution. Actual: concurrent runs can corrupt each other's cross-decode input/output, and a hung `vpxdec` can hang the entire soak indefinitely while leaving temp files behind.

## Fix

Use a unique per-run temporary directory or create-new paths scoped to the process, clean them up on all exits, close child stdin, and enforce a timeout with kill-on-timeout for the `vpxdec` process.

## Notes

`RVP9-BUG-KILN-00037` tracks analogous temp-file and unbounded subprocess risks in harness diagnostics. This bug tracks the benchmark soak harness.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Both halves of the observation are addressed at the root, and both meet the strict bar rather
than a cosmetic one. Uniqueness is genuine, not a renamed constant:
`temp_dir()/{prefix}-{pid}-{atomic seq}` created with `create_dir`, whose fail-if-exists
semantics drive a retry, so two concurrent soaks at the same iteration index cannot collide
(`crates/rvp9-bench/src/bin/soak.rs:53-75`). The wait is a real deadline with `kill()` + `wait()`
and an `Err` on expiry, not a longer sleep (`:83-124`, `:464-486`). Cleanup now happens on every
return path via `Drop`, including error returns, where the old code only removed files after a
successful `.output()`.
`tests::cross_decode_temp_dirs_are_isolated_and_cleaned` and
`tests::cross_decoder_timeout_kills_child` both observed passing; the timeout test spawns the
test binary as its own child, has it sleep 60s, and proves the runner returns a "timed out" `Err`
in under 5s -- it would hang forever against the old `.output()` code.
Exit path: a timeout returns `Err(String)` from `run_one_iteration`, which `main` matches at
`:595-604` and turns into `std::process::exit(2)`.
Residuals noted, none blocking: the timeout test never confirms the child is actually gone;
`Drop`-based cleanup does not survive a hard kill of soak itself (though unique names mean stale
dirs cannot corrupt a later run); and `VPXDEC_TIMEOUT` is a fixed 30s with no flag.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
