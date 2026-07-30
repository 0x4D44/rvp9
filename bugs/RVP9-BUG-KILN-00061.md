# RVP9-BUG-KILN-00061 - soak cross-decode uses shared temp files and an unbounded vpxdec wait

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/soak
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T190353Z-p998-n865634000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00061-run-fix-20260730T190353Z-p998-n865634000-c1
- **Owner base:** 2b4d7317eaf9859ee436c025c54d28e1a18b36ca
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T19:03:53Z
- **Owner until:** 2026-07-30T21:03:53Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`)

## Observation

The soak harness cross-decode leg writes deterministic filenames in the shared system temp directory and waits for `vpxdec` without a timeout.

`crates/rvp9-bench/src/bin/soak.rs:388`-`397` builds `rvp9_soak_iter{iter_idx}.ivf` and `.yuv` under `std::env::temp_dir()`. Two concurrent soak runs with the same iteration index can truncate, overwrite, read, or remove each other's files. `crates/rvp9-bench/src/bin/soak.rs:397`-`402` then runs `Command::new(vpxdec).output()` with no timeout or kill path; cleanup happens only after that call returns at `crates/rvp9-bench/src/bin/soak.rs:403`-`404`.

Expected: soak's optional external decoder leg should use isolated per-run temp paths and bounded child-process execution. Actual: concurrent runs can corrupt each other's cross-decode input/output, and a hung `vpxdec` can hang the entire soak indefinitely while leaving temp files behind.

## Fix

Use a unique per-run temporary directory or create-new paths scoped to the process, clean them up on all exits, close child stdin, and enforce a timeout with kill-on-timeout for the `vpxdec` process.

## Notes

`RVP9-BUG-KILN-00037` tracks analogous temp-file and unbounded subprocess risks in harness diagnostics. This bug tracks the benchmark soak harness.
