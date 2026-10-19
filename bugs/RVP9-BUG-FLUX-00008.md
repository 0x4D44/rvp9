# RVP9-BUG-FLUX-00008 — release core suite expects a debug-only assertion panic

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** tests/core mv-search
- **Raised:** 2026-08-07T11:41:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T054150Z-b2c37fe1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00008-run-verify-20261019T054150Z-b2c37fe1
- **Owner base:** abb4ebb97cda585ecd9989dc65bcc3e4ee057b27
- **Owner fingerprint:** sha256:3608f2a9ed10a261bdfc83704e53cbca69fc46ceb94f8cc4baa1d4b4ee5bd2b8
- **Owner since:** 2026-10-19T05:41:50Z
- **Owner until:** 2026-10-19T09:41:50Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-07T11:41:04Z, raised via `deltic bugs new`) -> Fixed (2026-08-12T14:56:56Z, deltic:auto role=fix run=fix-20260812T145350Z-p84799-n787877000-c1 branch=task/bug-RVP9-BUG-FLUX-00008-run-fix-20260812T145350Z-p84799-n787877000-c1 code=f7fd77ab gate=manual)

## Observation

Observation: cargo test -p rvp9-core --lib --release runs 1,659 tests but exits 101 because mv_search::sub_pixel::tests::sub_pixel_search_unsupported_block_size_panics_in_debug does not panic. The test carries #[should_panic] while its production trigger is a debug assertion, which release builds disable. Expected: the test is gated to debug-assertion builds or otherwise has a release-valid oracle, so both supported test profiles report their real result. Actual: 1,658 tests pass and this one test fails only under --release; the normal debug suite passes all 1,659.

## Fix

<unfixed — raised only>

## Notes
