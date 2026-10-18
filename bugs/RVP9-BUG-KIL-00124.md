# RVP9-BUG-KIL-00124 — noise-sensitivity silently clamps out-of-range CLI values

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** cli/rvp9enc input validation
- **Raised:** 2026-08-22T20:44:04Z
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
- **State history:** Open (2026-08-22T20:44:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T05:58:51Z, deltic:auto role=fix run=fix-20261016T054814Z-adb58265 branch=task/bug-RVP9-BUG-KIL-00124-run-fix-20261016T054814Z-adb58265 code=45238e97d0fd9863485fe8a3e23599411b8dafbd gate=manual) -> Closed (2026-10-18T16:58:10Z, deltic:manual role=verify run=verify-20261018T165508Z-31a19242 verified_fix_run=fix-20261016T054814Z-adb58265 verdict=close)

## Observation

The --noise-sensitivity handler accepts any u32 and the config builder silently clamps values above the documented 0..=6 domain. The in-tree test comment says the parser should catch this, but no range check exists. Expected: use parse_range for 0..=6 while retaining the defensive builder clamp. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:482-488,1520-1523,1833-1841; no app or test was run.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T16:58:10Z, deltic:auto run=verify-20261018T165508Z-31a19242 verified_fix_run=fix-20261016T054814Z-adb58265 verdict=close)

Verifier note: Manual verify: Independent native fixed-build verification on 45238e97: baseline and restored run passed 2 unit tests plus 1 CLI integration. Reversing the range parser accepted 7, 99, and u32::MAX in both argument forms; retained assertion and CLI pre-I/O rejection failed as expected. Valid 0/1/3/6, default zero, builder clamp, help range, and output preservation passed. Exact source restored; scoped diff clean. Neighboring validation untouched.

## Notes
