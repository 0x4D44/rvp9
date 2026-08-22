# RVP9-BUG-KIL-00092 — Additional vector-dependent tests report green without fixtures

- **State:** Fixed
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
- **State history:** Open (2026-08-22T13:20:54Z, raised via `deltic bugs new`) -> Fixed (2026-08-22T13:33:14Z, deltic:auto role=fix run=fix-20260822T133005Z-22df6bc2 branch=task/bug-RVP9-BUG-KIL-00092-run-fix-20260822T133005Z-22df6bc2 code=e9f7adf9 gate=manual)

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

<unfixed — raised only>

## Notes

Apply the crate's existing `rvp9_conformance_vectors` visible-ignore pattern,
then assert each exact vector and sidecar exist when enabled. This is a residual
of the same false-green class addressed at different sites by
`RVP9-BUG-KILN-00063`; these four sites are not named or fixed by that record.
Confirmed statically; the tests were not run by this review pass.
