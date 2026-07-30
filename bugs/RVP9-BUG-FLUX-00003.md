# RVP9-BUG-FLUX-00003 — Optional-vector differential tests fail clean worktrees after skipping every absent fixture

- **State:** Fixed
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) → Fixed (2026-07-30, `ecb6d88`, fixed by OpenAI Codex)

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
