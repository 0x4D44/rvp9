# RVP9-BUG-FLU-00029 — Legal empty VP9 tile row panics decoder

- **State:** Fixed
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
- **State history:** Open (2026-08-21T14:18:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-21T22:16:29Z, deltic:auto role=fix run=fix-20260821T221548Z-53d0861b branch=task/bug-RVP9-BUG-FLU-00029-run-fix-20260821T221548Z-53d0861b code=c9316971e1393c2270f29673efb64aa5b1c49f8c gate=manual)

## Observation

A legal VP9 frame with more tile rows than superblock rows produces a leading tile whose MI row interval is empty. TileDecodeCtx::new_with_view asserts that tile_mi_rows is greater than zero and panics instead of performing zero block iterations as libvpx does. The decoder must accept the empty interval and continue without panic.

## Fix

<unfixed — raised only>

## Notes
