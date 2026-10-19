# RVP9-BUG-FLUX-00012 — AQ test keeps encoder all-target clippy red

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** encoder/AQ tests
- **Raised:** 2026-08-09T20:10:46Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-09T20:10:46Z, raised via `deltic bugs new`) -> Fixed (2026-08-09T22:52:08Z, deltic:auto role=fix run=fix-20260809T223544Z-p48471-n589157000-c1 branch=task/bug-RVP9-BUG-FLUX-00012-run-fix-20260809T223544Z-p48471-n589157000-c1 code=dfba89f133ca12b3da29dca56c4b30a8d949188e gate=manual) -> Closed (2026-10-19T06:40:03Z, deltic:manual role=verify run=verify-20261019T063420Z-3fcbfdae verified_fix_run=fix-20260809T223544Z-p48471-n589157000-c1 verdict=close)

## Observation

Observation: `cargo clippy -p rvp9-encoder --all-targets -- -D warnings` fails at `crates/rvp9-encoder/src/aq_cyclic_refresh.rs:989` with `clippy::manual_contains`. The same failure reproduces from a clean archive of commit f3a330c2, so it predates the denoiser worktree. Expected: the documented all-target clippy gate passes; actual: the AQ test uses `seg_map.iter().any(|&s| s == CR_SEGMENT_ID_BOOST)` where current clippy requires `seg_map.contains(&CR_SEGMENT_ID_BOOST)`.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-19T06:40:03Z, deltic:auto run=verify-20261019T063420Z-3fcbfdae verified_fix_run=fix-20260809T223544Z-p48471-n589157000-c1 verdict=close)

Verifier note: Manual verify: Independent verification: the ledger SHA dfba89f133ca12b3da29dca56c4b30a8d949188e is absent; matching landed fix 8296cc1178e0722963d85f5f6db38d37a4cf9b42 is an ancestor of origin/main and changes the exact AQ assertion from iter().any to contains. cargo fmt --all --check passed. cargo clippy -p rvp9-encoder --all-targets passed with existing clone_on_copy (frame_loop.rs:14265) and match_like_matches_macro (spatial_encoder.rs:1082) warnings, both recorded in wrk_docs/evidence/2026-10-15-release-readiness/recovered-validation-20261017/clippy/clippy.raw.txt. Fails-before reversal r...

## Notes
