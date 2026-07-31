# RVP9-BUG-KILN-00029 - Public decode_tile_inter can panic on short previous segment maps

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T180322Z-p10015-n825996000-c1 branch=task/bug-RVP9-BUG-KILN-00029-run-fix-20260730T180322Z-p10015-n825996000-c1 code=1fc00e3fdb1de4333b29088cf887757f3b5db3c8 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`crates/rvp9-decoder/src/lib.rs:50` publicly exports the `tile` module, and `decode_tile_inter` is public at `crates/rvp9-decoder/src/tile.rs:1707`. It accepts caller-supplied `prev_segment_ids` at `crates/rvp9-decoder/src/tile.rs:1718`.

The inter segment decode path passes that slice to `read_inter_segment_id` at `crates/rvp9-decoder/src/tile.rs:2215`. The core helper treats an empty slice as no previous map, but any non-empty short slice is indexed without a length check at `crates/rvp9-core/src/mode_info.rs:200`.

The full `Decoder::decode_frame` path avoids this by passing an empty slice unless `prev_segment_ids.len() == mi_rows * mi_cols` at `crates/rvp9-decoder/src/lib.rs:2997`, but direct safe callers of the public tile API do not get that guard.

## Fix

Open. Validate `prev_segment_ids.len() >= mi_stride * needed_rows` before any inter segment lookup, or make the low-level tile decode API crate-private and expose only invariant-preserving decoder entry points. Add a public-API panic regression with a non-empty short previous segment map.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. This is related in shape to `RVP9-BUG-KILN-00023`, but it is a different public API and a different unchecked input.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The guard runs at the top of the public `decode_tile_inter` before any decoding and returns a
typed `PreviousSegmentMapTooShort` (`crates/rvp9-decoder/src/tile.rs:1750`). The bound was
checked as tight and correct: the only reader is `read_inter_segment_id` at stride
`params.base.mi_cols`, and `x_mis_pre`/`y_mis_pre` are clipped to frame `mi_cols`/`mi_rows`, so
the max index is exactly `mi_cols * mi_row_end - 1` -- the value `required` demands.
`decode_tile_inter` is the only public fn taking `prev_segment_ids`.
`tile::tests::decode_tile_inter_rejects_short_previous_segment_map_without_panicking` observed
passing; it asserts the exact error variant, so it fails without the guard.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
