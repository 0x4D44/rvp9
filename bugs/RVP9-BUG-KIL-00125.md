# RVP9-BUG-KIL-00125 — Symmetric CLI fixtures make plane-order and geometry tests vacuous

- **State:** Closed
- **Priority:** Should
- **Severity:** Low
- **Area:** cli/tests
- **Raised:** 2026-08-22T20:44:05Z
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
- **State history:** Open (2026-08-22T20:44:05Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T00:25:07Z, deltic:auto role=fix run=fix-20261016T001910Z-7f27bff9 branch=task/bug-RVP9-BUG-KIL-00125-run-fix-20261016T001910Z-7f27bff9 code=be3d2fc020ab0f3164c740095f3e79bfa3d4a5b9 gate=manual) -> Closed (2026-10-18T17:02:41Z, deltic:manual role=verify run=verify-20261018T165858Z-9a4002b3 verified_fix_run=fix-20261016T001910Z-7f27bff9 verdict=close)

## Observation

The main CLI fixtures are square 64x64 and generate identical U and V samples. The YV12 swap oracle therefore passes even if U and V are never swapped, while width-height transpositions evade most size-only assertions. Expected: use distinct U and V patterns plus at least one rectangular successful roundtrip that asserts container and Y4M dimensions and plane boundaries. Confirmed statically at crates/rvp9-cli/tests/cli_arg_coverage.rs:33-35,59-75,1233-1280 and crates/rvp9-cli/tests/cli_roundtrip.rs:24-26,47-68; no test was run.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T17:02:41Z, deltic:auto run=verify-20261018T165858Z-9a4002b3 verified_fix_run=fix-20261016T001910Z-7f27bff9 verdict=close)

Verifier note: Manual verify: Independent verification on the fixed build: both retained tests passed for YV12 plane distinction and rectangular 128x64 geometry/plane boundaries. Reversing identical U/V fixture samples made both retained tests fail their plane-distinction assertion. Exact fixtures restored byte-for-byte; scoped diff clean. No production code changed. Coverage is synthetic CLI geometry/plane behavior.

## Notes
