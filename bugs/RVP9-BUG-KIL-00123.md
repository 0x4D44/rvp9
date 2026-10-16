# RVP9-BUG-KIL-00123 — Raw invalid geometry allocates a frame before encoder validation

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc raw input validation
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
- **State history:** Open (2026-08-22T20:44:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T03:04:10Z, deltic:auto role=fix run=fix-20261016T025112Z-4e320457 branch=task/bug-RVP9-BUG-KIL-00123-run-fix-20261016T025112Z-4e320457 code=85153ed163d255b45c858ab619dc0d7d85919fd5 gate=manual)

## Observation

RawFrameReader computes plane sizes and reads the first complete frame before StreamEncoder validates the supported dimension and pixel-count envelope. Dimensions just above the 16384 limit can therefore allocate and read hundreds of megabytes only to return InvalidConfig afterward. Expected: validate raw geometry against the shared encoder limits before opening or reading frame data. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:932-949,1335-1431 and crates/rvp9-encoder/src/rate_control/config.rs:239-240,280-288; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
