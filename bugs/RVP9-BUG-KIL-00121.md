# RVP9-BUG-KIL-00121 — Two-pass encoding silently disables requested lookahead and ALTREF

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc option contract
- **Raised:** 2026-08-22T20:44:03Z
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
- **State history:** Open (2026-08-22T20:44:03Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T04:38:18Z, deltic:auto role=fix run=fix-20261016T042702Z-c6486cec branch=task/bug-RVP9-BUG-KIL-00121-run-fix-20261016T042702Z-c6486cec code=17d64800b1f8c8490400bbbacde486908d40adcd gate=manual)

## Observation

rvp9enc accepts --lag-in-frames and --auto-alt-ref with --passes 2, but encode_two_pass forces lag_in_frames to zero in both passes. ALTREF construction requires at least one forward lookahead frame, so a nonzero ALTREF request becomes inert without warning. Expected: reject the combinations until delayed two-pass scheduling exists, or implement it. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:1532-1546 and crates/rvp9-encoder/src/frame_loop.rs:845-850,887-892,3166-3185; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
