# RVP9-BUG-KILN-00068 — Stage 4a source-drift detection is not pinned by its regression tests

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/stage4a validation
- **Raised:** 2026-07-31T10:23:06Z
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T161822Z-ebd3a48c
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00068-run-verify-20261018T161822Z-ebd3a48c
- **Owner base:** 6a067e483f70acf6f5969f8bd263c2040ee029f4
- **Owner fingerprint:** sha256:33135824f9573c64c90f31dd75f7e599d7e2e914e7ce38944d69de4608730e0d
- **Owner since:** 2026-10-18T16:18:22Z
- **Owner until:** 2026-10-18T20:18:22Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31T10:23:06Z, raised via `deltic bugs new` model=claude-opus-5@high) -> Fixed (2026-10-16T05:20:28Z, deltic:auto role=fix run=fix-20261016T051313Z-ea0592df branch=task/bug-RVP9-BUG-KILN-00068-run-fix-20261016T051313Z-ea0592df code=6166d9a23e1720051d803dbf3aa1d061643db89c gate=manual)

## Observation

Split from RVP9-BUG-KILN-00016 during its 2026-07-31 two-eyes verification.

KILN-00016's fix (219fa56d) is correctly wired: crates/rvp9-encoder/tests/common/stage4a_product.rs:1464-1476 re-hashes the corpus inside refresh_final_validation, which is called at :1404 - before the status_reasons.is_empty() -> PASS decision at :1414 - so detected drift genuinely blocks a PASS.

The regression tests do not protect that. crates/rvp9-encoder/tests/stage4a_source_drift.rs (final_source_hash_detects_drift_and_pinned_hash_mismatch, final_source_hash_accepts_unchanged_pinned_input) exercise only the extracted string formatter at stage4a_product.rs:1433. Delete the sha256_file wiring block at :1464 and both tests still pass green - so the bug KILN-00016 was raised about could be reintroduced with the suite fully green.

Secondary: report.source_sha256 is overwritten with the FINAL hash, so the emitted JSON report loses the run-start value that the generated reason string refers to.

Expected: a test that fails if the final re-hash is removed from refresh_final_validation.
Actual: the wiring is verified only by reading.

Confirmed by reading the current tree at 4abe0d8 + 3a7aba2; both named tests observed passing in cargo test --workspace.

## Fix

<unfixed — raised only>

## Notes
