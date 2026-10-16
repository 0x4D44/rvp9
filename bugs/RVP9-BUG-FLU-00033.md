# RVP9-BUG-FLU-00033 — encode-headtohead spends 60 seconds sampling load at every clip without a ceiling

- **State:** Open
- **Priority:** Should
- **Severity:** Low
- **Area:** bench/encode-headtohead
- **Raised:** 2026-09-23T02:49:38Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T015734Z-7c798c86
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00033-run-fix-20261016T015734Z-7c798c86
- **Owner base:** 8de37847926c6fe337839a75f22ebee1b8b18e4e
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T01:57:34Z
- **Owner until:** 2026-10-16T05:57:34Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-09-23T02:49:38Z, raised via `deltic bugs new --land`)

## Observation

On Crucible, encode-headtohead with no --max-load1 still calls current_load1 at preflight, before each clip rotation, and at completion. Windows current_load1 blocks for 60 seconds per call. A three-clip, four-rotation screen therefore spends about 14 minutes sampling load even though no load ceiling can reject the run; an otherwise adequate 900-second deadline expires before timing completes. Expected: relaxed-load screens should not block for full-minute samples at every clip when the ceiling is absent; record useful load metadata without making the screen mostly idle waiting.

Evidence fingerprint: `manual:v1:encode-headtohead-spends-60-seconds-sampling-lo-02dbf5066e3c0332`


## Fix

<unfixed — raised only>

## Notes
