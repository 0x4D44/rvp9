# RVP9-BUG-KILN-00058 - bench-compare passes empty or one-sided comparisons

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/compare
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T185058Z-p85362-n070097000-c1 branch=task/bug-RVP9-BUG-KILN-00058-run-fix-20260730T185058Z-p85362-n070097000-c1 code=7b3b6d9b209922889caacbd66f63929bce0a75c3 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`bench-compare` can exit 0 without comparing any baseline/candidate samples.

The CSV reader silently skips malformed rows, unsupported schemas, and parse failures at `crates/rvp9-bench/src/bin/bench-compare.rs:122`-`148`. If no rows remain, `crates/rvp9-bench/src/bin/bench-compare.rs:219`-`224` prints only a warning. If the requested timestamps match no groups, `crates/rvp9-bench/src/bin/bench-compare.rs:261`-`263` writes a "(no rows matched the timestamps)" table row. The process exits non-zero only when a significant regression is found at `crates/rvp9-bench/src/bin/bench-compare.rs:316`-`320`.

One-sided groups also pass as noise: `crates/rvp9-bench/src/bin/bench-compare.rs:267`-`276` computes medians and Mann-Whitney U even if either side's vector is empty; `crates/rvp9-bench/src/mwu.rs:146`-`153` returns p=1.0 for empty groups.

Expected: a comparison command should fail when it has no valid matched groups, or when any emitted group lacks samples from both baseline and candidate. Actual: wrong timestamps, stale schemas, or one-sided data can produce an exit-0 "NOISE" result.

## Fix

Validate that at least one group has both baseline and candidate samples, and fail if any requested group is one-sided. Treat malformed current-schema rows as errors rather than silently skipping them, while still allowing explicit future-schema ignores if needed.

## Notes

This is specific to the statistical comparison tool. It is separate from the soft decode-throughput ratchet's intentionally disabled pinned-host enforcement.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

All four symptoms are closed at their exact sites: `parse_csv` returns `Err` on any malformed
schema-v3 field (`crates/rvp9-bench/src/bin/bench-compare.rs:109-175`), no valid rows exits 1
(`:250`), no groups matched the timestamps exits 1 (`:279`), and any one-sided group exits 1
(`:286-296`). The "(no rows matched the timestamps)" placeholder markdown row is deleted rather
than left reachable, and the one-sided guard runs BEFORE the median/MWU loop, so the empty-group
p=1.0 path the observation cited is now unreachable from this binary.
`timestamps_without_matching_rows_fail`, `one_sided_group_fails`,
`malformed_current_schema_row_fails` and `paired_group_passes_while_future_schema_is_ignored`
all observed passing. The last is a negative control pinning that a future schema-4 row with
only 8 columns is still ignored -- the schema check is deliberately ordered before the
column-count check.
Over-strict check against real data: the tracked `bench/results/decode-headtohead.csv` has 72
schema-1 and 12 schema-2 rows correctly ignored, and all 657 schema-3 rows have 22 columns with
parseable fields. The new strictness does not break the tracked corpus.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
