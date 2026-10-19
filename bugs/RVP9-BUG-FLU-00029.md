# RVP9-BUG-FLU-00029 — Legal empty VP9 tile row panics decoder

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/tiles
- **Raised:** 2026-08-21T14:18:04Z
- **Discovery source:** Automation
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
- **State history:** Open (2026-08-21T14:18:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-21T22:16:29Z, deltic:auto role=fix run=fix-20260821T221548Z-53d0861b branch=task/bug-RVP9-BUG-FLU-00029-run-fix-20260821T221548Z-53d0861b code=c9316971e1393c2270f29673efb64aa5b1c49f8c gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260821T221548Z-53d0861b branch=task/bug-RVP9-BUG-FLU-00029-run-fix-20260821T221548Z-53d0861b code=dba7e48ddcff3ec3eb572fcf475c763b7d504e81 gate=manual provenance-only=true reconciled-by=Codex) -> Closed (2026-10-19T16:16:35Z, deltic:manual role=verify run=verify-20261019T161337Z-7faef3f3 verified_fix_run=fix-20260821T221548Z-53d0861b verdict=close)

## Observation

A legal VP9 frame with more tile rows than superblock rows produces a leading tile whose MI row interval is empty. TileDecodeCtx::new_with_view asserts that tile_mi_rows is greater than zero and panics instead of performing zero block iterations as libvpx does. The decoder must accept the empty interval and continue without panic.

## Fix

Historical fix provenance is reconciled below. Independent behavioral verification remains pending.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `dba7e48ddcff3ec3eb572fcf475c763b7d504e81` (fix(codec): harden decoder reconstruction boundaries).
The original reference `c9316971e1393c2270f29673efb64aa5b1c49f8c` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `411e3bf511a5f84c6c45bc736c17b1d2ddefdf10` contains the original reference and descends from this fixing commit.
Commit removes tile_mi_rows > 0 assertions from new_with_grid and new_with_view, retaining column validation. Adds frame_view_accepts_a_legal_empty_tile_row and tile_bounds_allow_leading_empty_row regression definitions. These are the precise recorded legal-empty-tile constructor and geometry repair, rather than unrelated changes also bundled in this wider commit.
Contemporary corroboration: wrk_journals/2026.08.20 - JRN - Stage 5 safety qualification.md:130 (Unit 1262 names bug29 and zero-row assertion repair; added by this commit); wrk_docs/2026.08.22 - DAILY - KILN-Windows - Repo state report.md:113 (old c9316971 identified as same wider decoder-boundary hardening unit).
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

### Verification summary (2026-10-19T16:16:35Z, deltic:auto run=verify-20261019T161337Z-7faef3f3 verified_fix_run=fix-20260821T221548Z-53d0861b verdict=close)

Verifier note: Manual verify: Current frame_view_accepts_a_legal_empty_tile_row and tile_bounds_allow_leading_empty_row tests pass (1 each). On fix commit dba7e48, restoring only the removed tile_mi_rows > 0 assertion in TileDecodeCtx::new_with_view made the selected frame-view test panic with “tile must span at least one 8×8 mode-info row”. Removed the mutation, confirmed a clean scratch diff, and reran both tests successfully. The empty row matches the legal tile geometry in the record.

## Notes
