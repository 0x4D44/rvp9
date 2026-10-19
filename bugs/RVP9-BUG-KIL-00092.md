# RVP9-BUG-KIL-00092 — Additional vector-dependent tests report green without fixtures

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tests/false-green
- **Raised:** 2026-08-22T13:20:54Z
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
- **State history:** Open (2026-08-22T13:20:54Z, raised via `deltic bugs new`) -> Fixed (2026-08-22T13:33:14Z, deltic:auto role=fix run=fix-20260822T133005Z-22df6bc2 branch=task/bug-RVP9-BUG-KIL-00092-run-fix-20260822T133005Z-22df6bc2 code=e9f7adf9 gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260822T133005Z-22df6bc2 branch=task/bug-RVP9-BUG-KIL-00092-run-fix-20260822T133005Z-22df6bc2 code=745a0814fd3184b1282480003d348718f3cc127e gate=manual provenance-only=true reconciled-by=Codex) -> Closed (2026-10-19T11:08:34Z, deltic:manual role=verify run=verify-20261019T110633Z-085c0d6d verified_fix_run=fix-20260822T133005Z-22df6bc2 verdict=close)

## Observation

Four vector-dependent regression tests return normally when their required
fixture is absent, so Cargo reports them as passed after they assert nothing:

- `crates/rvp9-harness/tests/multi_tile.rs:27-35` (three multi-tile tests);
- `crates/rvp9-harness/tests/lossless.rs:42-50`;
- `crates/rvp9-harness/tests/intra_only.rs:41-48`; and
- `crates/rvp9-harness/tests/buffer_pool_recycle.rs:40-48`.

Expected: absent external vectors are visibly ignored, while enabled tests
require their exact fixtures. Actual: multi-tile decoding, lossless/WHT,
IntraOnly/show-existing decoding, and buffer-pool recycling can all appear
green without running their acceptance logic.

## Fix

Historical fix provenance is reconciled below. Independent behavioral verification remains pending.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `745a0814fd3184b1282480003d348718f3cc127e` (test(harness): expose missing vector fixtures).
The original reference `e9f7adf9` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `554d598263f9f36fbb2eecfcf40bf0cfe4793602` has this fixing commit as its immediate parent.
Adds rvp9_conformance_vectors cfg_attr visible ignores to all six tests across buffer_pool_recycle, intra_only, lossless, multi_tile; removes exact missing-file early returns, leaving required vector/sidecar expect reads.
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

### Verification summary (2026-10-19T11:08:34Z, deltic:auto run=verify-20261019T110633Z-085c0d6d verified_fix_run=fix-20260822T133005Z-22df6bc2 verdict=close)

Verifier note: Manual verify: Verified against ancestor fix 745a0814fd3184b1282480003d348718f3cc127e on origin/main. The corpus build script only enables vector tests for .ivf/.webm files; this worktree has only incomplete .part files. All six affected tests across four integration binaries were visibly ignored. Fails-before: restoring the old missing-file early return and removing the ignore attribute for vp90_2_08_tile_1x2_decodes_after_e2 made the selected test print “skip” and report 1 passed without reaching decode assertions. Restoring the fix returned all six tests to visible ignored status. With vect...

## Notes

Apply the crate's existing `rvp9_conformance_vectors` visible-ignore pattern,
then assert each exact vector and sidecar exist when enabled. This is a residual
of the same false-green class addressed at different sites by
`RVP9-BUG-KILN-00063`; these four sites are not named or fixed by that record.
Confirmed statically; the tests were not run by this review pass.
