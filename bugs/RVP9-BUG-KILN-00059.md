# RVP9-BUG-KILN-00059 - bdrate reports success when BD-rate or CSV measurement failed

- **State:** Open
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`)

## Observation

The BD-rate binary can complete with exit 0 when the measurement failed.

`decode_and_psnr` returns `NaN` on decoder spawn, decoder exit, or PSNR failure at `crates/rvp9-bench/src/bin/bdrate.rs:118`-`135`. `main` detects any `NaN`, prints "BD-rate: SKIP", and keeps running with `bd = None` at `crates/rvp9-bench/src/bin/bdrate.rs:300`-`305`. It also treats "insufficient points" as `None` at `crates/rvp9-bench/src/bin/bdrate.rs:306`-`310`.

When `--csv` is set, `write_csv` records a blank `bdrate_pct,` summary for `None` at `crates/rvp9-bench/src/bin/bdrate.rs:345`-`351`. If the CSV write itself fails, `crates/rvp9-bench/src/bin/bdrate.rs:318`-`321` only prints a warning. The binary never exits non-zero for those failed measurement states, despite the file header documenting CSV output as intended for CI ingestion at `crates/rvp9-bench/src/bin/bdrate.rs:27`-`30`.

Expected: failed decode/PSNR, missing BD-rate, or failed CSV output should make the command fail. Actual: a CI or script caller can accept a skipped BD-rate sweep as successful.

## Fix

Exit non-zero when any PSNR point is `NaN`, when BD-rate cannot be computed, or when requested CSV output cannot be written. If a diagnostic mode wants best-effort CSV, make that an explicit flag.

## Notes

The Stage 4+ quality gates should consume a finite BD-rate or an explicit failure, not a blank summary row.
