# RVP9-BUG-FLU-00029 — Legal empty VP9 tile row panics decoder

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/tiles
- **Raised:** 2026-08-21T14:18:04Z
- **Discovery source:** Automation
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260821T221548Z-53d0861b
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00029-run-fix-20260821T221548Z-53d0861b
- **Owner base:** bd22f9f47f63b627813234569ff751657d392b4f
- **Owner fingerprint:** -
- **Owner since:** 2026-08-21T22:15:48Z
- **Owner until:** 2026-08-22T00:15:48Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-21T14:18:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

A legal VP9 frame with more tile rows than superblock rows produces a leading tile whose MI row interval is empty. TileDecodeCtx::new_with_view asserts that tile_mi_rows is greater than zero and panics instead of performing zero block iterations as libvpx does. The decoder must accept the empty interval and continue without panic.

## Fix

<unfixed — raised only>

## Notes
