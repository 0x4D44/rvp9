# RVP9-BUG-KILN-00034 - Parallel harness oracles can pass on dropped or sentinel failures

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/threaded-test-oracles
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T191322Z-p9226-n944487000-c1 branch=task/bug-RVP9-BUG-KILN-00034-run-fix-20260730T191322Z-p9226-n944487000-c1 code=465a6b5273c4ad78d6621ebbebb179fee8514323 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

Several parallel decoder harness tests can report success after losing the evidence they are supposed to compare.

`crates/rvp9-harness/tests/tile_threading.rs:45` drops `parse_packet` failures with `continue`. The named-vector checks also skip missing vectors without a final checked-count assertion at lines `221`, `274`, `319`, and `416`.

The LF differential tests encode failures as digest values. `crates/rvp9-harness/tests/lf_overlap_pool_diff.rs:45` and line `61` push `[0u8; 16]` on parse or decode failure, then compare the vector at line `121`. `crates/rvp9-harness/tests/column_wavefront_lf_diff.rs:38` and line `54` do the same, then compare at line `116`. If both sides fail the same way, the tests compare identical sentinel bytes as if real decoded MD5s matched.

## Fix

Open. Fail missing required vectors in focused tests, return parse/decode failures as errors or explicit outcomes instead of MD5 bytes, and assert expected frame counts before comparing digests.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. This is adjacent to `RVP9-BUG-KILN-00032`, but that bug covers real-world/decoder corpus oracle absence rather than threaded harness oracles and sentinel digest comparisons.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The sentinel is gone at the source, not papered over: `decode_vector_md5s`/`decode_md5s` now
return `Result` and propagate parse/decode failures instead of pushing `[0u8; 16]` or `None`, so
two identically-FAILING legs can no longer compare equal. `continue`-on-missing-vector became
`assert!(path.exists(), ...)`, `checked == 0` warnings became `assert_eq!(checked, <expected>)`,
and every leg's length is checked against the `.md5` ledger before digests are compared
(`crates/rvp9-harness/build.rs`, `tests/tile_threading.rs:36-70`,
`tests/lf_overlap_pool_diff.rs:38-73`, `tests/column_wavefront_lf_diff.rs:31-66`).
These tests are `#[ignore]`d without vectors, so for this verification the conformance vectors
were fetched: `conformance_at_n_threads_2`, `determinism_n1_eq_n2`, `multi_tile_n1_eq_n2`,
`multi_tile_dispatch_uses_worker_pool`, `multi_tile_run_to_run_determinism`,
`scratch_tile_decode_and_lf_match_serial` and
`column_wavefront_lf_matches_serial_on_multi_tile_vectors` all ran and passed.
The remaining test in that file, `parallel_tile_decode_reuses_isolated_scratch_buffers`, fails
for an unrelated reason now tracked as RVP9-BUG-KILN-00062.
Residual noted, not blocking: this is a structural fix with no dedicated regression test --
nothing would catch someone re-introducing a sentinel push; only review would.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
