# RVP9-BUG-FLU-00033 — encode-headtohead spends 60 seconds sampling load at every clip without a ceiling

- **State:** Fixed
- **Priority:** Should
- **Severity:** Low
- **Area:** bench/encode-headtohead
- **Raised:** 2026-09-23T02:49:38Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T173112Z-8f58d4bd
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00033-run-verify-20261018T173112Z-8f58d4bd
- **Owner base:** ce8a038f547e98507aa9377c93c87f9eee838fd4
- **Owner fingerprint:** sha256:33bd3d63232a58d6a6176d40ba0ff388fc518d6bb5ca5e181de1aa9508224bc4
- **Owner since:** 2026-10-18T17:31:12Z
- **Owner until:** 2026-10-18T21:31:12Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-09-23T02:49:38Z, raised via `deltic bugs new --land`) -> Fixed (2026-10-16T02:07:48Z, deltic:auto role=fix run=fix-20261016T015734Z-7c798c86 branch=task/bug-RVP9-BUG-FLU-00033-run-fix-20261016T015734Z-7c798c86 code=a436015d8ce317b690531f77dabb10726b529145 gate=manual)

## Observation

On Crucible, encode-headtohead with no --max-load1 still calls current_load1 at preflight, before each clip rotation, and at completion. Windows current_load1 blocks for 60 seconds per call. A three-clip, four-rotation screen therefore spends about 14 minutes sampling load even though no load ceiling can reject the run; an otherwise adequate 900-second deadline expires before timing completes. Expected: relaxed-load screens should not block for full-minute samples at every clip when the ceiling is absent; record useful load metadata without making the screen mostly idle waiting.

Evidence fingerprint: `manual:v1:encode-headtohead-spends-60-seconds-sampling-lo-02dbf5066e3c0332`


## Fix

<unfixed — raised only>

## Notes
