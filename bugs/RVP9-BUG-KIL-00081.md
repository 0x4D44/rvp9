# RVP9-BUG-KIL-00081 — TileMiView geometry can drive safe methods out of bounds

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/tile context API
- **Raised:** 2026-08-22T10:24:23Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T070616Z-5f97e55c
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00081-run-verify-20261019T070616Z-5f97e55c
- **Owner base:** 833caa2908f2f5224fb47657c5a9c46f8c8a289e
- **Owner fingerprint:** sha256:55d9a3ef3db613d4ce2b262ac4132c83893c389d92af4d272c1fbd124221e880
- **Owner since:** 2026-10-19T07:06:16Z
- **Owner until:** 2026-10-19T11:06:16Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T10:24:23Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-22T12:44:28Z, deltic:auto role=fix run=fix-20260822T123644Z-5dab34f7 branch=task/bug-RVP9-BUG-KIL-00081-run-fix-20260822T123644Z-5dab34f7 code=f91347d9 gate=manual)

## Observation

`TileMiView::new` is unsafe, but its documented obligations at `crates/rvp9-decoder/src/tile_ctx.rs:128-156` require only a live backing allocation, non-overlapping rectangles, exclusive mutable access, and tile-row ordering. They do not require `mi_stride` and `bounds` to describe cells inside `len`. More importantly, the safe `TileDecodeCtx::new_with_view` at `:276-295` accepts independent dimensions and offsets without checking them against the view.

A caller can satisfy the constructor's stated unsafe obligations for a one-cell allocation and valid one-cell view, then call safe `new_with_view` with a mismatched offset. Safe `mi_at(0, 0)` computes that offset at `:334-342`, checks it only with `debug_assert!` at `:343-347`, and dereferences the resulting raw pointer at `:348-350`. Release builds can therefore read beyond the allocation. Independently, safe `write_block` documents clamping at `:447-449`, but a start column beyond `view.mi_col_end` makes `col_end_grid - abs_col` underflow before `from_raw_parts_mut` at `:470-488`, producing an invalid slice and write.

Expected: satisfying every documented unsafe-constructor obligation must make all later safe methods memory-safe for every safe input. Actual: safe geometry and coordinate arguments can reach out-of-bounds raw-pointer access.

## Fix

Validate and encapsulate the complete view invariant. `TileMiView::new` must require and check/document non-zero stride, ordered bounds, every bound inside the `len`-sized grid, and overflow-free index geometry. `TileDecodeCtx` should derive dimensions and offsets from the view, or make an inconsistent construction impossible. `mi_at` and `write_block` also need release guards with checked arithmetic before pointer construction. Add malformed-geometry and out-of-range-write regressions under Miri. Estimated effort: medium.

## Notes

Confirmed independently by the security lens, the lead's source trace, and the standing devil's-advocate pass. This is distinct from closed `RVP9-BUG-KILN-00027`, which covered concurrent shared-reference/raw-write aliasing in the disabled loop-filter overlap path. No Miri run or test was performed in this static review pass.
