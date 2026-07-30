# RVP9-BUG-KILN-00030 - Public frame_pool_mut can break decoder ref-map invariants

- **State:** Fixed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T180902Z-p19587-n303409000-c1 branch=task/bug-RVP9-BUG-KILN-00030-run-fix-20260730T180902Z-p19587-n303409000-c1 code=230878e69d288694ce1ef67082b9e81081cb330b gate=manual)

## Observation

`Decoder` documents a private invariant tying `ref_frame_map` to the frame pool at `crates/rvp9-decoder/src/lib.rs:301`: every map entry naming a pool slot must pair with a non-zero pool refcount and an occupied pool slot. Production writes preserve that through the private `write_ref_slot` helper at `crates/rvp9-decoder/src/lib.rs:869`.

The decoder still exposes the mutable pool directly through public `frame_pool_mut` at `crates/rvp9-decoder/src/lib.rs:844`. Safe external callers can then call `DecoderFramePool::assign_slot`, `release_slot`, or `clear` at `crates/rvp9-core/src/frame.rs:1420`, `crates/rvp9-core/src/frame.rs:1434`, and `crates/rvp9-core/src/frame.rs:1446` without updating the private `ref_frame_map`.

That can desynchronize a valid decoder and make later valid frames fail with empty references or reuse stale/replaced pool slots.

## Fix

Open. Make `frame_pool_mut` crate/test-only, or replace it with narrow public operations that update `ref_frame_map` and pool refcounts together. Add a safe public-API regression that mutates the pool behind an established reference map and proves the decoder returns a controlled error or no longer exposes the invariant break.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. Current in-repo uses are tests, but the method is part of the public safe API.
