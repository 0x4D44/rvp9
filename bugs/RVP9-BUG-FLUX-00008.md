# RVP9-BUG-FLUX-00008 — release core suite expects a debug-only assertion panic

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** tests/core mv-search
- **Raised:** 2026-08-07T11:41:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260812T145350Z-p84799-n787877000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00008-run-fix-20260812T145350Z-p84799-n787877000-c1
- **Owner base:** f33ad5076fcd8357bb7aa4f489d9ec9d9ef9a566
- **Owner fingerprint:** -
- **Owner since:** 2026-08-12T14:53:50Z
- **Owner until:** 2026-08-12T16:53:50Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-07T11:41:04Z, raised via `deltic bugs new`)

## Observation

Observation: cargo test -p rvp9-core --lib --release runs 1,659 tests but exits 101 because mv_search::sub_pixel::tests::sub_pixel_search_unsupported_block_size_panics_in_debug does not panic. The test carries #[should_panic] while its production trigger is a debug assertion, which release builds disable. Expected: the test is gated to debug-assertion builds or otherwise has a release-valid oracle, so both supported test profiles report their real result. Actual: 1,658 tests pass and this one test fails only under --release; the normal debug suite passes all 1,659.

## Fix

<unfixed — raised only>

## Notes
