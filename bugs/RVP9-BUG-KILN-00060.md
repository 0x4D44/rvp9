# RVP9-BUG-KILN-00060 - soak --cross-decode can pass after disabling the vpxdec leg

- **State:** Open
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`)

## Observation

The soak harness accepts `--cross-decode` but can silently run without the cross-decoder oracle.

If `--cross-decode` is set and the configured `vpxdec` path is missing, `crates/rvp9-bench/src/bin/soak.rs:452`-`460` prints a warning and sets `cross_decode_path` to `None`. The main loop then calls `run_one_iteration(..., cross_decode_path.as_deref())` at `crates/rvp9-bench/src/bin/soak.rs:503`-`508`, so the requested vpxdec leg is skipped for the entire run.

Expected: `--cross-decode` should fail fast when the cross-decoder cannot be found. Actual: a long soak can report success without ever exercising the requested cross-decode check.

## Fix

Make missing `--vpxdec` under `--cross-decode` a startup error. If best-effort probing is still useful, put it behind a separate explicit flag that says the cross-decode leg is optional.

## Notes

This is a false-green oracle bug, distinct from the subprocess boundedness and temp-file bug in `RVP9-BUG-KILN-00061`.
