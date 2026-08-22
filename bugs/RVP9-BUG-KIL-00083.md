# RVP9-BUG-KIL-00083 — Panic-safety inter recovery test accepts decode failure

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/panic-safety tests
- **Raised:** 2026-08-22T10:24:37Z
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
- **State history:** Open (2026-08-22T10:24:37Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

`Decoder`'s public error-safety documentation says a valid inter frame after a mid-tile failure "also decodes" at `crates/rvp9-decoder/src/lib.rs:205-212`. The matching `mid_tile_failure_then_valid_inter_decodes` test calls the recovery decode at `crates/rvp9-decoder/tests/panic_safety.rs:385-390`, but its `Err(_)` arm at `:406-414` accepts every error except `EmptyInterRef`.

A regression that preserves the reference-map slot while making the promised recovery frame fail with `Bitstream`, `TileDecode`, or another error therefore leaves the test green. Its name, documentation, and success branch claim successful decode; its assertion proves only reference-map resolvability.

Expected: the test proves the public recovery statement, or the statement and test are narrowed to the weaker reference-map contract. Actual: the gate accepts failure while presenting itself as a positive decode oracle.

## Fix

Choose and pin one contract. If the recovery inter must decode, replace the catch-all error arm with an `Ok` assertion and verify output dimensions or pixels. If entropy-bank poisoning intentionally permits failure, rename the test and narrow the public documentation, then add a separate positive oracle for whatever reuse guarantee remains. Estimated effort: small.

## Notes

Confirmed by the test-coverage lens, the lead's direct documentation/test comparison, and the standing devil's-advocate pass. No test was run in this static review pass.
