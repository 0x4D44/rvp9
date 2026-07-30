# RVP9-BUG-KILN-00058 - bench-compare passes empty or one-sided comparisons

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/compare
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T185058Z-p85362-n070097000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00058-run-fix-20260730T185058Z-p85362-n070097000-c1
- **Owner base:** 25cd83b9107392ce8c15b7123d2066c7120e6638
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T18:50:58Z
- **Owner until:** 2026-07-30T20:50:58Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`)

## Observation

`bench-compare` can exit 0 without comparing any baseline/candidate samples.

The CSV reader silently skips malformed rows, unsupported schemas, and parse failures at `crates/rvp9-bench/src/bin/bench-compare.rs:122`-`148`. If no rows remain, `crates/rvp9-bench/src/bin/bench-compare.rs:219`-`224` prints only a warning. If the requested timestamps match no groups, `crates/rvp9-bench/src/bin/bench-compare.rs:261`-`263` writes a "(no rows matched the timestamps)" table row. The process exits non-zero only when a significant regression is found at `crates/rvp9-bench/src/bin/bench-compare.rs:316`-`320`.

One-sided groups also pass as noise: `crates/rvp9-bench/src/bin/bench-compare.rs:267`-`276` computes medians and Mann-Whitney U even if either side's vector is empty; `crates/rvp9-bench/src/mwu.rs:146`-`153` returns p=1.0 for empty groups.

Expected: a comparison command should fail when it has no valid matched groups, or when any emitted group lacks samples from both baseline and candidate. Actual: wrong timestamps, stale schemas, or one-sided data can produce an exit-0 "NOISE" result.

## Fix

Validate that at least one group has both baseline and candidate samples, and fail if any requested group is one-sided. Treat malformed current-schema rows as errors rather than silently skipping them, while still allowing explicit future-schema ignores if needed.

## Notes

This is specific to the statistical comparison tool. It is separate from the soft decode-throughput ratchet's intentionally disabled pinned-host enforcement.
