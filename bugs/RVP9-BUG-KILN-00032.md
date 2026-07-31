# RVP9-BUG-KILN-00032 - Decoder-facing corpus tests can pass without required oracle data

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/test-oracles
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T190851Z-p5148-n949635000-c1 branch=task/bug-RVP9-BUG-KILN-00032-run-fix-20260730T190851Z-p5148-n949635000-c1 code=d494313d7626e957521132ffa558d9f60b0f23b0 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5); residual split to RVP9-BUG-KILN-00063)

## Observation

Two decoder-facing corpus oracles can report success without proving pixel correctness.

First, `crates/rvp9-harness/tests/real_world.rs:103` says `.md5` ledgers are optional and line `105` converts `read_md5_file` errors to `None`. The test only compares visible-frame MD5s inside `if let Some(entries)` at `crates/rvp9-harness/tests/real_world.rs:133`, so a missing or unparsable ledger can leave `total_mismatched == 0` and pass while checking no visible-frame pixels.

Second, `crates/rvp9-decoder/tests/conformance_coverage.rs:69` documents that the decoder crate's per-vector coverage mirror skips when vectors are absent, and `crates/rvp9-decoder/tests/conformance_coverage.rs:76` returns silently if a vector is missing. In a fresh worktree without fetched vectors, these tests can pass while decoding zero official conformance frames.

This is related to the repo's known fetched-data caveat, but it is more specific: named decoder-facing tests can be green without the oracle data they imply.

## Fix

Open. Require valid MD5 ledgers for any real-world fixture selected by the test, fail on parse errors, and assert shown-frame counts match ledger entries. For decoder conformance coverage, add an explicit strict mode used by local gates or make absent vectors produce an ignored/informational result rather than a silent success. Document the non-strict contributor path separately.

## Notes

Confirmed by two skeptic passes in the 2026-07-07 decoder review; the third classified these as intentional/report-only. It is filed because silent oracle disappearance is the same class of maintenance risk as the existing false-green test bugs, and the fix loop needs it in `bugs/` to drain it.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Both cited paths are genuinely closed: the `.md5` ledger is no longer optional
(`read_md5_file(...).unwrap_or_else(|e| panic!(...))`) and a new per-fixture
`assert_eq!(shown_idx, md5_entries.len())` blocks a vacuous pass
(`crates/rvp9-harness/tests/real_world.rs:103`, `:166`); the `vec_*` silent `return` became an
`assert!`, and absent vectors now yield `#[ignore]` rather than silent green
(`crates/rvp9-decoder/build.rs`, `tests/conformance_coverage.rs:76`).
`real_world_corpus_decodes_bit_exact` observed passing, and with conformance vectors fetched for
this verification all 14 `vec_*` tests ran and passed (they report `ignored` without vectors).
Residual split to RVP9-BUG-KILN-00063: `real_world.rs:77-84` still returns green with a printed
note when zero fixtures are present -- the same false-green class, one screen above the line
that was fixed.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
