# RVP9-BUG-KIL-00124 — noise-sensitivity silently clamps out-of-range CLI values

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** cli/rvp9enc input validation
- **Raised:** 2026-08-22T20:44:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T165508Z-31a19242
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00124-run-verify-20261018T165508Z-31a19242
- **Owner base:** 5107bda8c7202f473fb5bee5c2beab937a61461b
- **Owner fingerprint:** sha256:9ed860dd98bf6c60c06528f03167d13e0537f27171d3a7ecc017e6a7a483a195
- **Owner since:** 2026-10-18T16:55:08Z
- **Owner until:** 2026-10-18T20:55:08Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T05:58:51Z, deltic:auto role=fix run=fix-20261016T054814Z-adb58265 branch=task/bug-RVP9-BUG-KIL-00124-run-fix-20261016T054814Z-adb58265 code=45238e97d0fd9863485fe8a3e23599411b8dafbd gate=manual)

## Observation

The --noise-sensitivity handler accepts any u32 and the config builder silently clamps values above the documented 0..=6 domain. The in-tree test comment says the parser should catch this, but no range check exists. Expected: use parse_range for 0..=6 while retaining the defensive builder clamp. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:482-488,1520-1523,1833-1841; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
