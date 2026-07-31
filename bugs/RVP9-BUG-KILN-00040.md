# RVP9-BUG-KILN-00040 - MD5 ledger parser ignores filename frame indices

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T193818Z-p32779-n183602000-c1 branch=task/bug-RVP9-BUG-KILN-00040-run-fix-20260730T193818Z-p32779-n183602000-c1 code=2a445fbedbcac4c204a13fec0559f6a0fa0f6bb6 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The MD5 ledger parser accepts rows with duplicated, skipped, or reordered filename frame indices.

`crates/rvp9-harness/src/md5file.rs:12` documents that the MD5 filename encodes a one-based frame index. `read_md5_file` parses `(w, h, _idx)` at line `198`, discards `_idx`, and assigns `packet_index: row_index` at line `207` instead. `Md5Entry` has no field for the original filename index at line `57`.

That means a ledger with filename ordinals `1, 1, 3`, `1, 3`, or `2, 1` can parse as a valid sequence as long as each row is syntactically valid. Downstream conformance checks then trust row order rather than the embedded frame identity the ledger format claims to carry.

## Fix

Open. Validate the parsed filename index against the expected shown-frame ordinal, reject duplicates/skips/reordering, and preserve the original index in diagnostics where useful.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. The same parser feeds multiple conformance and corpus oracles, so this is tracked separately from individual test false-green bugs.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The discarded `_idx` is now compared against `row_index + 1` and a mismatch returns a typed
`Md5FileError::BadFrameIndex` carrying line, expected, actual and filename -- exactly the
requested remedy (`crates/rvp9-harness/src/md5file.rs:227-235`).
`md5file::tests::read_md5_file_rejects_nonsequential_filename_indices` covers all three reported
shapes (1,1 / 1,3 / 2,1) and is observed passing; all three parse clean pre-fix, so it genuinely
fails without the fix.
Over-strict-guard check, done two ways. All 44 checked-in `.md5` ledgers under `tests/` were
machine-checked against the new rule: 43 are strictly 1-based sequential; the 44th is a bare
single-hash file read by its own inline parser, never by `read_md5_file`. And the risk that
could not be settled from a bare worktree -- whether the DOWNLOADED conformance ledgers are
1-based -- was settled by fetching them: with 15 official vectors and their `.md5` sidecars
present, all 14 `vec_*` tests, `conformance_runner_walks_all_vectors` and
`real_world_corpus_decodes_bit_exact` parse and pass.
Residual noted, not blocking: `scripts/encode-corpus.ps1` orders rows by `Sort-Object Name`,
which breaks lexicographically past 9999 frames -- that is now a hard error rather than silent
misordering.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
