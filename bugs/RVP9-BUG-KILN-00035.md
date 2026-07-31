# RVP9-BUG-KILN-00035 - F.0.2 block oracle never compares the libvpx and rvp9 legs

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** harness/f02-block-oracle
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-31, deltic:auto role=fix run=fix-20260730T230244Z-p58756-n105437000-c1 branch=task/bug-RVP9-BUG-KILN-00035-run-fix-20260730T230244Z-p58756-n105437000-c1 code=048d412dc6e2ed4c488a9ac11c72ee2667374b4a gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`f02-block-oracle` is described as a paired libvpx-vs-rvp9 oracle, but the implementation does not compare the two legs.

`crates/rvp9-harness/src/bin/f02_block_oracle.rs:5` says the tool is a "libvpx-vs-rvp9 picker-decision oracle", and line `14` advertises raw paired records. The implementation loads or parses the libvpx sidecar at lines `128` and `321`, then only writes per-frame count summaries at line `156` and rvp9 JSONL at line `183`. Normal completion reaches `Ok(())` at line `197` with no pairwise record comparison and no mismatch exit.

The result is that a divergent libvpx oracle leg can be present, parsed, and still never checked against the rvp9 records the tool emits.

## Fix

Open. Align records by frame/block key, compare the fields that define the oracle contract, emit paired diagnostics, and exit nonzero on mismatches unless an explicit report-only mode is selected.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. No exact existing F.0.2 oracle bug was found.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The legs are now aligned by `(frame_idx, mi_row, mi_col)` and seven live fields compared;
missing records and duplicate keys count as mismatches, so a dropped leg cannot vanish
(`crates/rvp9-harness/src/f02_schema.rs:310`, `:240`). `main` accumulates into
`strict_mismatches` and returns `Err` at `crates/rvp9-harness/src/bin/f02_block_oracle.rs:249`,
making the process exit non-zero; report-only is an explicit opt-in flag. A missing libvpx
sidecar is also fatal, so the comparison cannot be skipped by accident, and paired output goes
to a different filename than the rvp9 leg, so neither clobbers the other.
`f02_schema::tests::divergent_live_field_is_a_strict_oracle_failure`,
`::comparison_aligns_by_key_and_ignores_rate_placeholders`, `::missing_and_duplicate_keys_are_mismatches`
and `::paired_jsonl_contains_both_legs_and_mismatch_fields` all observed passing.
Residuals noted, not blocking: the four tests exercise the library comparator, not the binary's
wiring -- nothing tests that `main` actually calls `require_match`; and the module doc at
`f02_block_oracle.rs:15` still claims the rvp9 leg lands in `_raw.jsonl`, which now holds the
paired output.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
