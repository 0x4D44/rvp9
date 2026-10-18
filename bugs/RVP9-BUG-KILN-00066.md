# RVP9-BUG-KILN-00066 — decode-throughput default sweep still exits 0 after a failed fixture

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/decode-throughput
- **Raised:** 2026-07-31T10:22:37Z
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T165854Z-8724175c
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00066-run-verify-20261018T165854Z-8724175c
- **Owner base:** e3ca07aa934274ab2a10e0621c46c5221ccf5ec4
- **Owner fingerprint:** sha256:e8c2d8aeb73dbc84013b5e0a994dfdf63cea4c1d48fd965a413b4604809d3693
- **Owner since:** 2026-10-18T16:58:54Z
- **Owner until:** 2026-10-18T20:58:54Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31T10:22:37Z, raised via `deltic bugs new` model=claude-opus-5@high) -> Fixed (2026-10-16T05:30:54Z, deltic:auto role=fix run=fix-20261016T052048Z-d55a7572 branch=task/bug-RVP9-BUG-KILN-00066-run-fix-20261016T052048Z-d55a7572 code=a40334ca6e256b7f2ba651a3fc01097bc3e2e881 gate=manual)

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

## Notes
