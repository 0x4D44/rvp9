# RVP9-BUG-KILN-00016 - Stage 4a product validation misses ignored corpus drift during a run

- **State:** Closed
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`) -> Fixed (2026-07-31, deltic:auto role=fix run=fix-20260730T231507Z-p86647-n962624000-c1 branch=task/bug-RVP9-BUG-KILN-00016-run-fix-20260730T231507Z-p86647-n962624000-c1 code=219fa56d62b392ff768f655faef7b9f49e97a64d gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5); residual split to RVP9-BUG-KILN-00068)

## Observation

The Stage 4a product runner records the initial SHA-256 of `tests/corpus/foreman_cif.y4m` into `source_sha256` at `crates/rvp9-encoder/tests/common/stage4a_product.rs:567`. Final validation calls `refresh_final_validation` at `crates/rvp9-encoder/tests/common/stage4a_product.rs:1433`, which refreshes git and source-tree state. `validate_worktree` recomputes the ignored corpus hash into `allowed_ignored_inputs` at `crates/rvp9-encoder/tests/common/stage4a_product.rs:1494`, but no final check compares that hash to the initial `source_sha256`.

Because the corpus file is intentionally ignored and not covered by the tracked source-tree hash, it can change during a long product run without a status reason.

## Fix

Open. Re-hash `FOREMAN_REL` during final validation and compare it with the start-of-run `source_sha256` and expected pinned hash. Emit a dedicated changed-during-run status on mismatch.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The reported symptom is fixed and the wiring is correct: `refresh_final_validation`
re-hashes the corpus (`crates/rvp9-encoder/tests/common/stage4a_product.rs:1464-1476`) and is
called at `:1404`, before the `status_reasons.is_empty() -> PASS` decision at `:1414`, so
detected drift genuinely blocks a PASS.
`final_source_hash_detects_drift_and_pinned_hash_mismatch` and
`final_source_hash_accepts_unchanged_pinned_input` both observed passing.
Residual split to RVP9-BUG-KILN-00068: those two tests exercise only the extracted string
formatter. Delete the `sha256_file` wiring block at `:1464` and both still pass green, so the
fix is not actually pinned by its own regression coverage.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
