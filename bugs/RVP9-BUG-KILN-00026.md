# RVP9-BUG-KILN-00026 - Tile entropy decode accepts EOF-overread payloads

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** decoder/tile entropy
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T162528Z-p98885-n775251000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00026-run-fix-20260730T162528Z-p98885-n775251000-c1
- **Owner base:** dc76dede3939839efdac78c30323c1b4bd232b3a
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T16:25:28Z
- **Owner until:** 2026-07-30T18:25:28Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`)

## Observation

`BoolReader` preserves libvpx EOF semantics: reads that cross EOF return valid-looking bits, and callers must check `has_error()` after decoding. The core reader documents that at `crates/rvp9-core/src/entropy/bool_reader.rs:8`, and compressed-header parsing checks it before accepting a header at `crates/rvp9-core/src/bitstream/compressed_header.rs:546`.

Tile decode does not do the same. `decode_tile_intra` creates a tile `BoolReader` at `crates/rvp9-decoder/src/tile.rs:429` and returns `Ok(decode_ctx)` at `crates/rvp9-decoder/src/tile.rs:518` without checking `reader.has_error()`. `decode_tile_inter` has the same shape at `crates/rvp9-decoder/src/tile.rs:1738` and `crates/rvp9-decoder/src/tile.rs:1796`.

Malformed or truncated tile payloads can therefore decode using EOF sentinel bits and publish output/reference state instead of returning a tile decode error.

## Fix

Open. Add a tile-level truncated/reader-overread error and check `reader.has_error()` before returning success from both tile decode paths. Add malformed intra/inter tile regressions that reach the bool-reader EOF sentinel without failing an earlier structural check.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. Existing truncation panic-safety tests assert that some truncated streams return `Err`, but they do not prove every tile EOF overread is rejected at the post-decode `BoolReader::has_error()` boundary.
