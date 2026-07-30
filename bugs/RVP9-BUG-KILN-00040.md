# RVP9-BUG-KILN-00040 - MD5 ledger parser ignores filename frame indices

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/md5-ledger
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T193818Z-p32779-n183602000-c1 branch=task/bug-RVP9-BUG-KILN-00040-run-fix-20260730T193818Z-p32779-n183602000-c1 code=2a445fbedbcac4c204a13fec0559f6a0fa0f6bb6 gate=manual)

## Observation

The MD5 ledger parser accepts rows with duplicated, skipped, or reordered filename frame indices.

`crates/rvp9-harness/src/md5file.rs:12` documents that the MD5 filename encodes a one-based frame index. `read_md5_file` parses `(w, h, _idx)` at line `198`, discards `_idx`, and assigns `packet_index: row_index` at line `207` instead. `Md5Entry` has no field for the original filename index at line `57`.

That means a ledger with filename ordinals `1, 1, 3`, `1, 3`, or `2, 1` can parse as a valid sequence as long as each row is syntactically valid. Downstream conformance checks then trust row order rather than the embedded frame identity the ledger format claims to carry.

## Fix

Open. Validate the parsed filename index against the expected shown-frame ordinal, reject duplicates/skips/reordering, and preserve the original index in diagnostics where useful.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. The same parser feeds multiple conformance and corpus oracles, so this is tracked separately from individual test false-green bugs.
