# RVP9-BUG-KILN-00029 - Public decode_tile_inter can panic on short previous segment maps

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/tile public API
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

`crates/rvp9-decoder/src/lib.rs:50` publicly exports the `tile` module, and `decode_tile_inter` is public at `crates/rvp9-decoder/src/tile.rs:1707`. It accepts caller-supplied `prev_segment_ids` at `crates/rvp9-decoder/src/tile.rs:1718`.

The inter segment decode path passes that slice to `read_inter_segment_id` at `crates/rvp9-decoder/src/tile.rs:2215`. The core helper treats an empty slice as no previous map, but any non-empty short slice is indexed without a length check at `crates/rvp9-core/src/mode_info.rs:200`.

The full `Decoder::decode_frame` path avoids this by passing an empty slice unless `prev_segment_ids.len() == mi_rows * mi_cols` at `crates/rvp9-decoder/src/lib.rs:2997`, but direct safe callers of the public tile API do not get that guard.

## Fix

Open. Validate `prev_segment_ids.len() >= mi_stride * needed_rows` before any inter segment lookup, or make the low-level tile decode API crate-private and expose only invariant-preserving decoder entry points. Add a public-API panic regression with a non-empty short previous segment map.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. This is related in shape to `RVP9-BUG-KILN-00023`, but it is a different public API and a different unchecked input.
