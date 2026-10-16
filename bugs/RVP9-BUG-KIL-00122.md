# RVP9-BUG-KIL-00122 — Two-pass temporal SVC discards first-pass allocations

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc two-pass temporal SVC
- **Raised:** 2026-08-22T20:44:03Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T003414Z-3df93afe
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00122-run-fix-20261016T003414Z-3df93afe
- **Owner base:** ca431d610472ed52477052b5f7c11508c8842635
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T00:34:14Z
- **Owner until:** 2026-10-16T04:34:14Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:03Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

rvp9enc permits --passes 2 with --temporal-layers 2 or 3. encode_two_pass attaches TwoPassState only to the stream controller, but temporal SVC replaces it before Q selection with a per-layer controller whose two_pass field is empty, causing the two-pass picker to fall back to average bandwidth. Expected: reject two-pass temporal SVC until per-layer first-pass state is implemented. Confirmed statically at crates/rvp9-encoder/src/frame_loop.rs:843-851,3777-3795 and rate_control/two_pass.rs:255-264; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
