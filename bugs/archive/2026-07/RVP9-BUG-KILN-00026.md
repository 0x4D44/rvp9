# RVP9-BUG-KILN-00026 - Tile entropy decode accepts EOF-overread payloads

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** decoder/tile entropy
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T162528Z-p98885-n775251000-c1 branch=task/bug-RVP9-BUG-KILN-00026-run-fix-20260730T162528Z-p98885-n775251000-c1 code=1b8df1914ad885a03f52c8e70b1d6b81b627c106 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`BoolReader` preserves libvpx EOF semantics: reads that cross EOF return valid-looking bits, and callers must check `has_error()` after decoding. The core reader documents that at `crates/rvp9-core/src/entropy/bool_reader.rs:8`, and compressed-header parsing checks it before accepting a header at `crates/rvp9-core/src/bitstream/compressed_header.rs:546`.

Tile decode does not do the same. `decode_tile_intra` creates a tile `BoolReader` at `crates/rvp9-decoder/src/tile.rs:429` and returns `Ok(decode_ctx)` at `crates/rvp9-decoder/src/tile.rs:518` without checking `reader.has_error()`. `decode_tile_inter` has the same shape at `crates/rvp9-decoder/src/tile.rs:1738` and `crates/rvp9-decoder/src/tile.rs:1796`.

Malformed or truncated tile payloads can therefore decode using EOF sentinel bits and publish output/reference state instead of returning a tile decode error.

## Fix

Open. Add a tile-level truncated/reader-overread error and check `reader.has_error()` before returning success from both tile decode paths. Add malformed intra/inter tile regressions that reach the bool-reader EOF sentinel without failing an earlier structural check.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. Existing truncation panic-safety tests assert that some truncated streams return `Err`, but they do not prove every tile EOF overread is rejected at the post-decode `BoolReader::has_error()` boundary.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The check is exactly the one the observation named, at exactly the two return sites named
(`crates/rvp9-decoder/src/tile.rs:537-539`, `:1832-1834`), using the same `BoolReader::has_error()`
the compressed-header path already uses. The predicate mirrors libvpx `vpx_reader_has_error`,
and libvpx performs the identical post-tile check, so this is parity rather than an rvp9
invention.
`tile::tests::truncated_intra_tile_payload_is_rejected_after_entropy_decode` and
`::truncated_inter_tile_payload_is_rejected_after_entropy_decode` both observed passing.
Over-strict-guard check -- the risk here is rejecting VALID tightly-packed streams. The live
canary is `profile1_vectors_decode_bit_exact` (real checked-in streams through both tile paths),
which passes, plus the 14 `vec_*` conformance tests and `conformance_runner_walks_all_vectors`,
all observed passing with vectors fetched. No false positive on real bitstreams.
Residual noted, not blocking: the fix had to repair a synthetic fixture
(`build_minimal_intra_only_with_fc_idx`, 4 trailing zero bytes -> 512) that was relying on EOF
sentinel bits, which changes the input of several unrelated intra-only tests in that module.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
