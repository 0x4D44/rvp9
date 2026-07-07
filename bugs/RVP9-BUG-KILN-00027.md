# RVP9-BUG-KILN-00027 - LF-overlap reads an immutable MI-grid slice while tile workers mutate the same grid

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** decoder/loopfilter overlap threading
- **Raised:** 2026-07-07
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`)

## Observation

The LF-overlap path builds loop-filter row inputs from immutable references to `frame_mi_grid`: `frame_mi_grid.as_slice()` is stored in `RowMaskInputs` at `crates/rvp9-decoder/src/lib.rs:1972` and in `Non420Inputs` at `crates/rvp9-decoder/src/lib.rs:1987`. The LF job is then dispatched before the tile decode jobs complete at `crates/rvp9-decoder/src/lib.rs:2015`.

The same frame MI grid is also handed to tile workers as raw mutable `TileMiView` pointers. The decoder creates those views from `frame_mi_grid.as_mut_ptr()` at `crates/rvp9-decoder/src/lib.rs:1866`, and `TileDecodeCtx::set_block` writes back through `from_raw_parts_mut` at `crates/rvp9-decoder/src/tile_ctx.rs:463`. The loop-filter mask builder later reads the immutable slice at `crates/rvp9-core/src/loopfilter/apply.rs:306`.

The row publication protocol may prevent a logical read-before-write race, but it does not make an outstanding immutable Rust slice compatible with concurrent raw mutable writes to the same allocation. This is a separate aliasing defect from `RVP9-BUG-KILN-00024`, which tracks overlapping full-plane mutable pixel slices.

## Fix

Open. Do not keep `&[Option<ModeInfo>]` over an in-flight mutable MI grid. Either disable LF overlap until tile decode joins, represent the MI slab through an `UnsafeCell`/raw-pointer abstraction with documented row publication, or publish immutable per-row LF mask inputs after decode owns each row.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. Cross-link: `RVP9-BUG-KILN-00024` covers the adjacent pixel-plane aliasing problem, but not this MI-grid shared-reference/raw-mutation alias.
