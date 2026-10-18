# RVP9-BUG-KIL-00122 — Two-pass temporal SVC discards first-pass allocations

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc two-pass temporal SVC
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
- **State history:** Open (2026-08-22T20:44:03Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T00:40:34Z, deltic:auto role=fix run=fix-20261016T003414Z-3df93afe branch=task/bug-RVP9-BUG-KIL-00122-run-fix-20261016T003414Z-3df93afe code=38cd5c8edb90ccae39503e86cecc8b2ecc5139a1 gate=manual) -> Closed (2026-10-18T18:09:03Z, deltic:manual role=verify run=verify-20261018T180234Z-c4e1f69f verified_fix_run=fix-20261016T003414Z-3df93afe verdict=close)

## Observation

rvp9enc permits --passes 2 with --temporal-layers 2 or 3. encode_two_pass attaches TwoPassState only to the stream controller, but temporal SVC replaces it before Q selection with a per-layer controller whose two_pass field is empty, causing the two-pass picker to fall back to average bandwidth. Expected: reject two-pass temporal SVC until per-layer first-pass state is implemented. Confirmed statically at crates/rvp9-encoder/src/frame_loop.rs:843-851,3777-3795 and rate_control/two_pass.rs:255-264; no app or test was run.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T18:09:03Z, deltic:auto run=verify-20261018T180234Z-c4e1f69f verified_fix_run=fix-20261016T003414Z-3df93afe verdict=close)

Verifier note: Manual verify: Independent verification passed: retained encoder/API and CLI regressions each pass on the integrated fix, both fail their own assertion when the temporal-layer guard is weakened from >1 to >3, and both pass after exact source restoration. API covers 2/3 layers with empty and three-frame inputs; CLI covers 2/3 layers and confirms no diagnostic or output creation on the mutant. Focused tests only; no full suite. Left stale Fix placeholder and Notes unchanged.

## Notes
