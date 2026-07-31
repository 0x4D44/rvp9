# RVP9-BUG-FLUX-00003 — Optional-vector differential tests fail clean worktrees after skipping every absent fixture

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness differential tests
- **Raised:** 2026-07-30
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
- **Attempts:** fix=1, doubt=0, indeterminate=0
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) → Fixed (2026-07-30, `ecb6d88`, fixed by OpenAI Codex) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

Observed in a clean task worktree on aarch64-apple-darwin with Rust 1.95.0:
`cargo test --workspace --locked` reaches the harness differential suites,
skips each absent optional conformance vector, then
`column_wavefront_lf_matches_serial_on_multi_tile_vectors` fails because
`checked` remains zero. `lf_overlap_pool_matches_serial` has the same terminal
assertion. Expected: the documented workspace gate passes without gitignored
downloaded corpora, while each differential assertion and path-engagement check
still runs whenever a listed vector is present. Actual: an empty optional corpus
makes the normal local gate fail.

## Fix

Commit `ecb6d88` treats an empty optional vector set as a skipped oracle in both
differential suites. A present vector still runs every serial/parallel frame
comparison, and the LF-overlap suite still requires its path-engagement counter
to advance. Both tests pass with no downloaded vectors and pass again with the
official `vp90-2-08-tile_1x2.webm`; `cargo test --workspace --locked` passes in
the clean checkout state.

## Notes

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Resolved at HEAD, but NOT by the mechanism this record names -- correcting that here.
The recorded fix `ecb6d88` (eprintln-and-continue) was superseded by `465a6b5` and `a915c7f`,
which replaced it with a `build.rs`-driven `#[cfg_attr(not(rvp9_conformance_vectors), ignore)]`.
That is strictly better: an absent corpus ignores visibly, and a present corpus makes every
vector mandatory (`assert!(path.exists())` plus `assert_eq!(checked, ...)`).
Verified both ways on this host: vectorless the two tests report `ignored`; with vectors
fetched `column_wavefront_lf_matches_serial_on_multi_tile_vectors` and
`scratch_tile_decode_and_lf_match_serial` both run and pass.
Two further record corrections: the test formerly called `lf_overlap_pool_matches_serial` is
now `scratch_tile_decode_and_lf_match_serial`, and the Fix text above claims the LF-overlap
suite requires its engagement counter to ADVANCE -- at HEAD it asserts the opposite
(`overlap_frames_post == overlap_frames_pre`), because the overlap path is deliberately
disabled (see RVP9-BUG-KILN-00027 / RVP9-BUG-KILN-00067).

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
