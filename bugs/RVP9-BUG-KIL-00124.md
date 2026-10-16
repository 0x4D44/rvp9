# RVP9-BUG-KIL-00124 — noise-sensitivity silently clamps out-of-range CLI values

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** cli/rvp9enc input validation
- **Raised:** 2026-08-22T20:44:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T054814Z-adb58265
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00124-run-fix-20261016T054814Z-adb58265
- **Owner base:** e6a73bc3decb90ed945a026a41e4f5780ff174c2
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T05:48:14Z
- **Owner until:** 2026-10-16T09:48:14Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

The --noise-sensitivity handler accepts any u32 and the config builder silently clamps values above the documented 0..=6 domain. The in-tree test comment says the parser should catch this, but no range check exists. Expected: use parse_range for 0..=6 while retaining the defensive builder clamp. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:482-488,1520-1523,1833-1841; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
