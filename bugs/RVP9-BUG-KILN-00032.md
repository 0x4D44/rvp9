# RVP9-BUG-KILN-00032 - Decoder-facing corpus tests can pass without required oracle data

- **State:** Open
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`)

## Observation

Two decoder-facing corpus oracles can report success without proving pixel correctness.

First, `crates/rvp9-harness/tests/real_world.rs:103` says `.md5` ledgers are optional and line `105` converts `read_md5_file` errors to `None`. The test only compares visible-frame MD5s inside `if let Some(entries)` at `crates/rvp9-harness/tests/real_world.rs:133`, so a missing or unparsable ledger can leave `total_mismatched == 0` and pass while checking no visible-frame pixels.

Second, `crates/rvp9-decoder/tests/conformance_coverage.rs:69` documents that the decoder crate's per-vector coverage mirror skips when vectors are absent, and `crates/rvp9-decoder/tests/conformance_coverage.rs:76` returns silently if a vector is missing. In a fresh worktree without fetched vectors, these tests can pass while decoding zero official conformance frames.

This is related to the repo's known fetched-data caveat, but it is more specific: named decoder-facing tests can be green without the oracle data they imply.

## Fix

Open. Require valid MD5 ledgers for any real-world fixture selected by the test, fail on parse errors, and assert shown-frame counts match ledger entries. For decoder conformance coverage, add an explicit strict mode used by local gates or make absent vectors produce an ignored/informational result rather than a silent success. Document the non-strict contributor path separately.

## Notes

Confirmed by two skeptic passes in the 2026-07-07 decoder review; the third classified these as intentional/report-only. It is filed because silent oracle disappearance is the same class of maintenance risk as the existing false-green test bugs, and the fix loop needs it in `bugs/` to drain it.
