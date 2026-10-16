# RVP9-BUG-KIL-00123 — Raw invalid geometry allocates a frame before encoder validation

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc raw input validation
- **Raised:** 2026-08-22T20:44:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T025112Z-4e320457
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00123-run-fix-20261016T025112Z-4e320457
- **Owner base:** c0da4e56ef0c665e1dbf5c26318e61d20ab116b6
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T02:51:12Z
- **Owner until:** 2026-10-16T06:51:12Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

RawFrameReader computes plane sizes and reads the first complete frame before StreamEncoder validates the supported dimension and pixel-count envelope. Dimensions just above the 16384 limit can therefore allocate and read hundreds of megabytes only to return InvalidConfig afterward. Expected: validate raw geometry against the shared encoder limits before opening or reading frame data. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:932-949,1335-1431 and crates/rvp9-encoder/src/rate_control/config.rs:239-240,280-288; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
