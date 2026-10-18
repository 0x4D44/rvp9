# RVP9-BUG-KILN-00067 — LF-overlap MI-grid aliasing construct remains in-tree behind a disabled const

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/loopfilter overlap
- **Raised:** 2026-07-31T10:22:52Z
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T161119Z-25634be8
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00067-run-verify-20261018T161119Z-25634be8
- **Owner base:** efa9b36f24ac3556cfaf96ab64c8bf141d6675c5
- **Owner fingerprint:** sha256:d7e3fb492a1e47bd44452562f5b79194bcf933d0f07adebfe8c0f8d645ca4ecd
- **Owner since:** 2026-10-18T16:11:19Z
- **Owner until:** 2026-10-18T20:11:19Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31T10:22:52Z, raised via `deltic bugs new` model=claude-opus-5@high) -> Fixed (2026-10-16T04:54:46Z, deltic:auto role=fix run=fix-20261016T044757Z-fc395d57 branch=task/bug-RVP9-BUG-KILN-00067-run-fix-20261016T044757Z-fc395d57 code=fdf0d7f665b05291e70167bd74a5c60f3e5468ef gate=manual)

## Observation

Split from RVP9-BUG-KILN-00027 during its 2026-07-31 two-eyes verification.

KILN-00027 (LF-overlap reads an immutable MI-grid slice while tile workers mutate the same grid) was resolved by DISABLING the path, not by repairing it: crates/rvp9-decoder/src/lib.rs:100 sets OUTPUT_FRAME_LF_OVERLAP_ALIAS_SAFE = false, and the RowMaskInputs { mi_grid: frame_mi_grid.as_slice() } constructions at lib.rs:2032/2046 and :3416/3430 sit behind that compile-time-false conjunction. The bug's own Fix section listed disabling as an acceptable option, so the closure is legitimate - the unsound construction is unreachable at runtime today.

What remains: the unsound code is retained verbatim, and flipping one const re-arms it. There is no compile-time or test-time guard tying the const to the aliasing precondition, and the only test that exercises the path (crates/rvp9-harness/tests/lf_overlap_pool_diff.rs, scratch_tile_decode_and_lf_match_serial) merely asserts the overlap counter did NOT move - a proxy for 'the feature is off', not a proof the design is sound. It is also #[cfg_attr(not(rvp9_conformance_vectors), ignore)], so it does not run in a vectorless worktree at all.

Two further consequences: the LF-overlap performance path is now permanently off, and _num_workers in apply_loop_filter_to_frame_column_wavefront_dispatch is a dead parameter.

Expected: either the aliasing is repaired (workers publish their MI rectangle before any immutable whole-grid slice is taken) and the path re-enabled, or the dead construction is removed so it cannot be re-armed by flipping a bool.
Actual: unsound code retained, dormant, one const away from live.

Confirmed by reading the current tree at 4abe0d8 + 3a7aba2.

## Fix

<unfixed — raised only>

## Notes
