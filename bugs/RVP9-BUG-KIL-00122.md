# RVP9-BUG-KIL-00122 — Two-pass temporal SVC discards first-pass allocations

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc two-pass temporal SVC
- **Raised:** 2026-08-22T20:44:03Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T180234Z-c4e1f69f
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00122-run-verify-20261018T180234Z-c4e1f69f
- **Owner base:** 7c7d0b13d93b9a89f0fc729a502ae563e20559ed
- **Owner fingerprint:** sha256:324c6a9f17792e66b8ef8cde8d65b4dcc198f3658c0c57b1d7e5b1c8105ce08f
- **Owner since:** 2026-10-18T18:02:34Z
- **Owner until:** 2026-10-18T22:02:34Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:03Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T00:40:34Z, deltic:auto role=fix run=fix-20261016T003414Z-3df93afe branch=task/bug-RVP9-BUG-KIL-00122-run-fix-20261016T003414Z-3df93afe code=38cd5c8edb90ccae39503e86cecc8b2ecc5139a1 gate=manual)

## Observation

rvp9enc permits --passes 2 with --temporal-layers 2 or 3. encode_two_pass attaches TwoPassState only to the stream controller, but temporal SVC replaces it before Q selection with a per-layer controller whose two_pass field is empty, causing the two-pass picker to fall back to average bandwidth. Expected: reject two-pass temporal SVC until per-layer first-pass state is implemented. Confirmed statically at crates/rvp9-encoder/src/frame_loop.rs:843-851,3777-3795 and rate_control/two_pass.rs:255-264; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
