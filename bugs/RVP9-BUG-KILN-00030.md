# RVP9-BUG-KILN-00030 - Public frame_pool_mut can break decoder ref-map invariants

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/frame pool public API
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T180902Z-p19587-n303409000-c1 branch=task/bug-RVP9-BUG-KILN-00030-run-fix-20260730T180902Z-p19587-n303409000-c1 code=230878e69d288694ce1ef67082b9e81081cb330b gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`Decoder` documents a private invariant tying `ref_frame_map` to the frame pool at `crates/rvp9-decoder/src/lib.rs:301`: every map entry naming a pool slot must pair with a non-zero pool refcount and an occupied pool slot. Production writes preserve that through the private `write_ref_slot` helper at `crates/rvp9-decoder/src/lib.rs:869`.

The decoder still exposes the mutable pool directly through public `frame_pool_mut` at `crates/rvp9-decoder/src/lib.rs:844`. Safe external callers can then call `DecoderFramePool::assign_slot`, `release_slot`, or `clear` at `crates/rvp9-core/src/frame.rs:1420`, `crates/rvp9-core/src/frame.rs:1434`, and `crates/rvp9-core/src/frame.rs:1446` without updating the private `ref_frame_map`.

That can desynchronize a valid decoder and make later valid frames fail with empty references or reuse stale/replaced pool slots.

## Fix

Open. Make `frame_pool_mut` crate/test-only, or replace it with narrow public operations that update `ref_frame_map` and pool refcounts together. Add a safe public-API regression that mutates the pool behind an established reference map and proves the decoder returns a controlled error or no longer exposes the invariant break.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. Current in-repo uses are tests, but the method is part of the public safe API.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

`frame_pool_mut` is now `#[cfg(test)]` and private (`crates/rvp9-decoder/src/lib.rs:888`); the
`frame_pool` field is private and `set_ref_slot` is `#[cfg(test)] pub(crate)`. No remaining
public path hands out a `&mut DecoderFramePool` -- every surviving `frame_pool_mut` caller is
inside the crate's own `mod tests`.
Gated by a `compile_fail` doctest on `Decoder::frame_pool` (`lib.rs:858-864`), observed passing.
Residual noted, not blocking: a `compile_fail` doctest with no error-code annotation passes on
ANY compile failure, so it is a weak gate -- but it is a real one against re-publishing the
accessor.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
