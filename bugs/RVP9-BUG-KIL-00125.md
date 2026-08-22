# RVP9-BUG-KIL-00125 — Symmetric CLI fixtures make plane-order and geometry tests vacuous

- **State:** Open
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
- **State history:** Open (2026-08-22T20:44:05Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

The main CLI fixtures are square 64x64 and generate identical U and V samples. The YV12 swap oracle therefore passes even if U and V are never swapped, while width-height transpositions evade most size-only assertions. Expected: use distinct U and V patterns plus at least one rectangular successful roundtrip that asserts container and Y4M dimensions and plane boundaries. Confirmed statically at crates/rvp9-cli/tests/cli_arg_coverage.rs:33-35,59-75,1233-1280 and crates/rvp9-cli/tests/cli_roundtrip.rs:24-26,47-68; no test was run.

## Fix

<unfixed — raised only>

## Notes
