# RVP9-BUG-FLUX-00008 — release core suite expects a debug-only assertion panic

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** tests/core mv-search
- **Raised:** 2026-08-07T11:41:04Z
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
- **State history:** Open (2026-08-07T11:41:04Z, raised via `deltic bugs new`) -> Fixed (2026-08-12T14:56:56Z, deltic:auto role=fix run=fix-20260812T145350Z-p84799-n787877000-c1 branch=task/bug-RVP9-BUG-FLUX-00008-run-fix-20260812T145350Z-p84799-n787877000-c1 code=f7fd77ab gate=manual) -> Closed (2026-10-19T06:01:21Z, deltic:manual role=verify run=verify-20261019T054150Z-b2c37fe1 verified_fix_run=fix-20260812T145350Z-p84799-n787877000-c1 verdict=close)

## Observation

Observation: cargo test -p rvp9-core --lib --release runs 1,659 tests but exits 101 because mv_search::sub_pixel::tests::sub_pixel_search_unsupported_block_size_panics_in_debug does not panic. The test carries #[should_panic] while its production trigger is a debug assertion, which release builds disable. Expected: the test is gated to debug-assertion builds or otherwise has a release-valid oracle, so both supported test profiles report their real result. Actual: 1,658 tests pass and this one test fails only under --release; the normal debug suite passes all 1,659.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-19T06:01:21Z, deltic:auto run=verify-20261019T054150Z-b2c37fe1 verified_fix_run=fix-20260812T145350Z-p84799-n787877000-c1 verdict=close)

Verifier note: Manual verify: Independent manual verification by Codex. The recorded fix SHA f7fd77ab is absent from origin/main, but the matching landed test change c9ab18be0f247d6a1347d2377e12b39b49d815c6 is an ancestor of origin/main. The source has a debug_assert! for unsupported subpixel variance, compiled out in release; the should_panic test is now ignored when debug_assertions are disabled. Original release observation rerun: cargo test -p rvp9-core --lib --release passed 1737, ignored 3, failed 0. Focused debug test selected 1 and passed via should_panic. Fails-before: removed only the cfg_attr rele...

## Notes
