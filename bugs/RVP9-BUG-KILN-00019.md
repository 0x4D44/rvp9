# RVP9-BUG-KILN-00019 - Non-ignored encoder tests have false-green oracles

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/test-oracles
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`) -> Fixed (2026-07-31, deltic:auto role=fix run=fix-20260730T231934Z-p94033-n610643000-c1 branch=task/bug-RVP9-BUG-KILN-00019-run-fix-20260730T231934Z-p94033-n610643000-c1 code=0333c772565054bcb3094b5d96c49c42dfadaf39 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

Several non-ignored encoder tests can pass without proving the property named by the test:

- `w_lag_in_frames_nonzero_rejects_sync_encode_frame` at `crates/rvp9-encoder/src/frame_loop.rs:9556` stores an error from `expect_err`, then computes `matches!(err, EncodeError::InvalidConfig(_))` at `crates/rvp9-encoder/src/frame_loop.rs:9585` and discards the boolean. The test fails if `encode_frame` succeeds, but passes for any error variant.
- `phase2_newmv_reuse_fires` at `crates/rvp9-encoder/tests/phase2_reuse_positive_control.rs:55` returns successfully when `foreman_cif` is absent at line 57.
- `a21_ml_partition_breakout_fires_and_decodes` at `crates/rvp9-encoder/tests/w_ml_partition.rs:224` returns successfully when `testsrc_720p30.y4m` is missing or unreadable at lines 226 and 240.

These tests are not ignored, so a normal local test run can report green while the intended oracle did not run or did not assert the expected condition.

## Fix

Open. Wrap the discarded `matches!` call in an assertion. For the corpus-dependent tests, use deterministic synthetic fixtures, make missing required corpus a hard failure, or mark the tests ignored/informational so they do not count as green proof in normal runs.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

All three observed false-green sites are fixed and each was checked in the current tree: the
discarded `matches!` is now inside an `assert!` (`crates/rvp9-encoder/src/frame_loop.rs:9656`),
and both corpus-dependent proofs are `#[ignore]`d with the silent `return`-on-missing replaced
by `expect`, plus an added `assert_eq!(frames.len(), N_FRAMES)` so an `--ignored` run on a
truncated corpus fails instead of half-proving.
`frame_loop::tests::w_lag_in_frames_nonzero_rejects_sync_encode_frame` observed passing;
`phase2_newmv_reuse_fires` and `a21_ml_partition_breakout_fires_and_decodes` observed as
`ignored`, which is the intended post-fix state.
Residual noted, not blocking: the remedy trades false green for no coverage -- both proofs now
never run in the normal gate. This record explicitly sanctioned that option. A follow-up to give
each a synthetic fixture is recorded in `scratchpad.md`.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
