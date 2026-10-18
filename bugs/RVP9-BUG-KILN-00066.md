# RVP9-BUG-KILN-00066 — decode-throughput default sweep still exits 0 after a failed fixture

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/decode-throughput
- **Raised:** 2026-07-31T10:22:37Z
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
- **State history:** Open (2026-07-31T10:22:37Z, raised via `deltic bugs new` model=claude-opus-5@high) -> Fixed (2026-10-16T05:30:54Z, deltic:auto role=fix run=fix-20261016T052048Z-d55a7572 branch=task/bug-RVP9-BUG-KILN-00066-run-fix-20261016T052048Z-d55a7572 code=a40334ca6e256b7f2ba651a3fc01097bc3e2e881 gate=manual) -> Closed (2026-10-18T17:02:12Z, deltic:manual role=verify run=verify-20261018T165854Z-8724175c verified_fix_run=fix-20261016T052048Z-d55a7572 verdict=close)

## Observation

Split from RVP9-BUG-KILN-00057 during its 2026-07-31 two-eyes verification.

KILN-00057's fix (013ddfd4) made two of the three symptoms in its Observation fatal: an empty result set and a CSV-append failure both now return ExitCode::FAILURE from main. The third symptom it named - crates/rvp9-bench/src/bin/decode-throughput.rs:176-178, a partial run where some fixtures failed - is unchanged: the binary prints '(one or more fixtures failed...)' and returns ExitCode::SUCCESS provided at least one other fixture recorded a row.

That closes the loop with RVP9-BUG-KILN-00056, which made a malformed packet fail its fixture's measurement. Under an explicit --fixture the failure collapses to results.is_empty() and is now fatal, but in the default multi-fixture sweep a fixture that fails to measure is still reported only on stderr, and the process exits 0 with partial data written to the tracked ratchet CSV.

Expected: a sweep with any failed fixture exits non-zero, or records the failure in the CSV so the ratchet cannot silently narrow.
Actual: exit 0 with a partial dataset.

KILN-00057's own Fix section flagged this limb as optional, which is why it was closed rather than reopened.

Confirmed by reading the current tree at 4abe0d8 + 3a7aba2.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T17:02:12Z, deltic:auto run=verify-20261018T165854Z-8724175c verified_fix_run=fix-20261016T052048Z-d55a7572 verdict=close)

Verifier note: Manual verify: Independent Codex verification on 165cc9a2912ede3f (fix a40334ca ancestor): cargo test -p rvp9-bench --test decode_throughput_cli default_sweep selected 2 and passed before/after. Real generated VP9 mixed default sweep exits nonzero while retaining one positive-frame CSV row; complete sweep exits0 with all4 rows. Removing only had_error failure return selected mixed regression1 and failed assertion partial default sweep unexpectedly exited successfully (one row plus malformed/missing fixture errors); restored exact bytes finally, scoped diff and worktree status clean. Commands b...

## Notes
