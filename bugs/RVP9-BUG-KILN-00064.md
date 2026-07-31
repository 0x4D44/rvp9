# RVP9-BUG-KILN-00064 — f02_block_oracle still shares temp files and waits unbounded for vpxenc

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/f02 oracle
- **Raised:** 2026-07-31T10:22:15Z
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
- **State history:** Open (2026-07-31T10:22:15Z, raised via `deltic bugs new` model=claude-opus-5@high)

## Observation

Split from RVP9-BUG-KILN-00037 during its 2026-07-31 two-eyes verification.

KILN-00037's Observation named TWO sites carrying the shared-temp-file / unbounded-child-wait defect: crates/rvp9-harness/src/bin/e1_block_stats.rs and crates/rvp9-harness/src/bin/f02_block_oracle.rs:302. The accepted fix (068c152a) rebuilt e1_block_stats with a ScopedTempDir (pid + atomic sequence, create_dir fail-and-retry, Drop cleanup) and a run_with_timeout that closes stdin, polls to a 30s deadline, then kills and reaps. f02_block_oracle was left untouched.

At trunk 4abe0d8 + 3a7aba2 that call is now at crates/rvp9-harness/src/bin/f02_block_oracle.rs:363 and is still a plain cmd.output()? with no timeout and no .stdin(Stdio::null()). A hung or stdin-reading patched vpxenc strands the oracle run indefinitely.

Expected: f02_block_oracle uses the same bounded-child and per-run temp-directory discipline e1_block_stats now has.
Actual: unchanged.

Confirmed by reading the current tree; not reproduced at runtime (needs a patched vpxenc).

## Fix

<unfixed — raised only>

## Notes
