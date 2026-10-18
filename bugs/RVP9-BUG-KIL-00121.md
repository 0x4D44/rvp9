# RVP9-BUG-KIL-00121 — Two-pass encoding silently disables requested lookahead and ALTREF

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc option contract
- **Raised:** 2026-08-22T20:44:03Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T164147Z-cc410bb6
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00121-run-verify-20261018T164147Z-cc410bb6
- **Owner base:** fa7184edc531ff9d528828e1824a8acbb8700eb8
- **Owner fingerprint:** sha256:60259bfcfb93507393758964507f8cf65108cabfc23592b66d18f925e5d9896b
- **Owner since:** 2026-10-18T16:41:47Z
- **Owner until:** 2026-10-18T20:41:47Z
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
