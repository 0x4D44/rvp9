# RVP9-BUG-KILN-00016 - Stage 4a product validation misses ignored corpus drift during a run

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/stage4a product validation
- **Raised:** 2026-07-05
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`)

## Observation

The Stage 4a product runner records the initial SHA-256 of `tests/corpus/foreman_cif.y4m` into `source_sha256` at `crates/rvp9-encoder/tests/common/stage4a_product.rs:567`. Final validation calls `refresh_final_validation` at `crates/rvp9-encoder/tests/common/stage4a_product.rs:1433`, which refreshes git and source-tree state. `validate_worktree` recomputes the ignored corpus hash into `allowed_ignored_inputs` at `crates/rvp9-encoder/tests/common/stage4a_product.rs:1494`, but no final check compares that hash to the initial `source_sha256`.

Because the corpus file is intentionally ignored and not covered by the tracked source-tree hash, it can change during a long product run without a status reason.

## Fix

Open. Re-hash `FOREMAN_REL` during final validation and compare it with the start-of-run `source_sha256` and expected pinned hash. Emit a dedicated changed-during-run status on mismatch.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.
