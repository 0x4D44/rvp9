# RVP9-BUG-KILN-00057 - decode-throughput exits successfully when measurement or CSV recording fails

- **State:** Fixed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T184605Z-p81009-n041713000-c1 branch=task/bug-RVP9-BUG-KILN-00057-run-fix-20260730T184605Z-p81009-n041713000-c1 code=013ddfd44077f8d2f39a94cd744771a906d1f36c gate=manual)

## Observation

`decode-throughput` returns exit 0 after failed measurement states that should make the command unusable as a recorder.

If no fixture produces timed samples, `crates/rvp9-bench/src/bin/decode-throughput.rs:152`-`155` prints "nothing to record" and returns from `main`. If `throughput::append_csv` fails, `crates/rvp9-bench/src/bin/decode-throughput.rs:163`-`172` prints the write failure and also returns. A partial run with at least one failed fixture only prints a warning at `crates/rvp9-bench/src/bin/decode-throughput.rs:176`-`178`.

Expected: a benchmark command should exit non-zero when it produced no measurement rows or could not write the requested CSV. Actual: callers can treat a missing or unrecorded measurement as success.

## Fix

Exit non-zero when `results.is_empty()` or CSV append fails. Consider also returning non-zero for fixture errors when the user passed an explicit `--fixture`; the default multi-fixture sweep can still choose whether partial data is acceptable.

## Notes

The ratchet test may remain the performance gate, but the measurement binary should still distinguish "measurement recorded" from "no data / no CSV".
