# RVP9-BUG-FLUX-00012 — AQ test keeps encoder all-target clippy red

- **State:** Fixed
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
- **State history:** Open (2026-08-09T20:10:46Z, raised via `deltic bugs new`) -> Fixed (2026-08-09T22:52:08Z, deltic:auto role=fix run=fix-20260809T223544Z-p48471-n589157000-c1 branch=task/bug-RVP9-BUG-FLUX-00012-run-fix-20260809T223544Z-p48471-n589157000-c1 code=dfba89f133ca12b3da29dca56c4b30a8d949188e gate=manual)

## Observation

Observation: `cargo clippy -p rvp9-encoder --all-targets -- -D warnings` fails at `crates/rvp9-encoder/src/aq_cyclic_refresh.rs:989` with `clippy::manual_contains`. The same failure reproduces from a clean archive of commit f3a330c2, so it predates the denoiser worktree. Expected: the documented all-target clippy gate passes; actual: the AQ test uses `seg_map.iter().any(|&s| s == CR_SEGMENT_ID_BOOST)` where current clippy requires `seg_map.contains(&CR_SEGMENT_ID_BOOST)`.

## Fix

<unfixed — raised only>

## Notes
