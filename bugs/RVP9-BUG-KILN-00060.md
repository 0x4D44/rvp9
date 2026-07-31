# RVP9-BUG-KILN-00060 - soak --cross-decode can pass after disabling the vpxdec leg

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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T185939Z-p94676-n911942000-c1 branch=task/bug-RVP9-BUG-KILN-00060-run-fix-20260730T185939Z-p94676-n911942000-c1 code=9cad2792d4eb411e68601df6aa2df8a590a1a03b gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The soak harness accepts `--cross-decode` but can silently run without the cross-decoder oracle.

If `--cross-decode` is set and the configured `vpxdec` path is missing, `crates/rvp9-bench/src/bin/soak.rs:452`-`460` prints a warning and sets `cross_decode_path` to `None`. The main loop then calls `run_one_iteration(..., cross_decode_path.as_deref())` at `crates/rvp9-bench/src/bin/soak.rs:503`-`508`, so the requested vpxdec leg is skipped for the entire run.

Expected: `--cross-decode` should fail fast when the cross-decoder cannot be found. Actual: a long soak can report success without ever exercising the requested cross-decode check.

## Fix

Make missing `--vpxdec` under `--cross-decode` a startup error. If best-effort probing is still useful, put it behind a separate explicit flag that says the cross-decode leg is optional.

## Notes

This is a false-green oracle bug, distinct from the subprocess boundedness and temp-file bug in `RVP9-BUG-KILN-00061`.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The WARN-and-degrade branch is deleted outright and the check moved to the very top of `main`,
ahead of `build_source_list()`, so it fails before any work: a missing vpxdec under
`--cross-decode` prints "error: --cross-decode requires vpxdec at ..." and calls
`std::process::exit(2)` (`crates/rvp9-bench/src/bin/soak.rs:520-528`).
`requested_cross_decode_requires_existing_vpxdec` observed passing; it runs the real soak binary
with a non-existent vpxdec path and asserts BOTH a non-zero exit and the exact error substring --
the substring assertion is what stops it passing spuriously on an unrelated arg-parse failure.
Against the old code soak printed WARN and completed with exit 0.
Residual noted, not blocking: the guard is `is_file()` only, so a path that exists but is not
executable satisfies it and the run fails later at spawn.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
