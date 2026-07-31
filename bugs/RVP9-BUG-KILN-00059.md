# RVP9-BUG-KILN-00059 - bdrate reports success when BD-rate or CSV measurement failed

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/bdrate
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T185527Z-p89787-n764622000-c1 branch=task/bug-RVP9-BUG-KILN-00059-run-fix-20260730T185527Z-p89787-n764622000-c1 code=a1b4da090b9a47551dd4721b1e2a928217a2a500 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The BD-rate binary can complete with exit 0 when the measurement failed.

`decode_and_psnr` returns `NaN` on decoder spawn, decoder exit, or PSNR failure at `crates/rvp9-bench/src/bin/bdrate.rs:118`-`135`. `main` detects any `NaN`, prints "BD-rate: SKIP", and keeps running with `bd = None` at `crates/rvp9-bench/src/bin/bdrate.rs:300`-`305`. It also treats "insufficient points" as `None` at `crates/rvp9-bench/src/bin/bdrate.rs:306`-`310`.

When `--csv` is set, `write_csv` records a blank `bdrate_pct,` summary for `None` at `crates/rvp9-bench/src/bin/bdrate.rs:345`-`351`. If the CSV write itself fails, `crates/rvp9-bench/src/bin/bdrate.rs:318`-`321` only prints a warning. The binary never exits non-zero for those failed measurement states, despite the file header documenting CSV output as intended for CI ingestion at `crates/rvp9-bench/src/bin/bdrate.rs:27`-`30`.

Expected: failed decode/PSNR, missing BD-rate, or failed CSV output should make the command fail. Actual: a CI or script caller can accept a skipped BD-rate sweep as successful.

## Fix

Exit non-zero when any PSNR point is `NaN`, when BD-rate cannot be computed, or when requested CSV output cannot be written. If a diagnostic mode wants best-effort CSV, make that an explicit flag.

## Notes

The Stage 4+ quality gates should consume a finite BD-rate or an explicit failure, not a blank summary row.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The "BD-rate: SKIP ... bd = None" branch is deleted. A non-finite PSNR point or an uncomputable
BD-rate now returns `ExitCode::FAILURE` from `main` before any CSV is written, and the blank
`bdrate_pct,` row is structurally impossible because `write_csv` takes the `Option` and errors on
`None` before creating the file (`crates/rvp9-bench/src/bin/bdrate.rs:194`, `:318`, `:335`,
`:349-357`). The check also widened from `is_nan()` to `!is_finite()`, catching infinities. Both
failure returns were traced by reading the full `main` -- no `let _ =` or `.ok()`.
`tests::csv_writer_rejects_missing_bdrate_without_creating_output` and
`tests::measured_bdrate_rejects_failed_psnr_and_insufficient_points` observed passing; neither
compiles or passes against the old predicates.
Residuals noted, neither blocking: the tests are unit-level only -- nothing executes the binary,
so a regression that reverted `main`'s exit code while leaving `measured_bd_rate` intact would go
uncaught (an end-to-end test needs external vpxenc/vpxdec). And two adjacent, out-of-scope
weaknesses in the same file -- `encode_one` uses `.status().expect()` plus `assert!`, and
`decode_and_psnr` writes a shared temp Y4M under `temp_dir()` with no timeout, the same class as
RVP9-BUG-KILN-00061 -- are recorded in `scratchpad.md`.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
