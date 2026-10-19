# RVP9-BUG-KIL-00083 — Panic-safety inter recovery test accepts decode failure

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/panic-safety tests
- **Raised:** 2026-08-22T10:24:37Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T110415Z-0a4d6703
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00083-run-verify-20261019T110415Z-0a4d6703
- **Owner base:** 206268f1bbb990f59b137d195a0c42429d93568f
- **Owner fingerprint:** sha256:a5744ec2cfcf197f2e9861fd44feeb1377c70d522f17290a41b03dacfa6adb25
- **Owner since:** 2026-10-19T11:04:15Z
- **Owner until:** 2026-10-19T15:04:15Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T10:24:37Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-22T13:13:44Z, deltic:auto role=fix run=fix-20260822T131153Z-3ad51f2d branch=task/bug-RVP9-BUG-KIL-00083-run-fix-20260822T131153Z-3ad51f2d code=267e292b gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260822T131153Z-3ad51f2d branch=task/bug-RVP9-BUG-KIL-00083-run-fix-20260822T131153Z-3ad51f2d code=f274787175bf93421861efe348861a3646659cec gate=manual provenance-only=true reconciled-by=Codex)

## Observation

`Decoder`'s public error-safety documentation says a valid inter frame after a mid-tile failure "also decodes" at `crates/rvp9-decoder/src/lib.rs:205-212`. The matching `mid_tile_failure_then_valid_inter_decodes` test calls the recovery decode at `crates/rvp9-decoder/tests/panic_safety.rs:385-390`, but its `Err(_)` arm at `:406-414` accepts every error except `EmptyInterRef`.

A regression that preserves the reference-map slot while making the promised recovery frame fail with `Bitstream`, `TileDecode`, or another error therefore leaves the test green. Its name, documentation, and success branch claim successful decode; its assertion proves only reference-map resolvability.

Expected: the test proves the public recovery statement, or the statement and test are narrowed to the weaker reference-map contract. Actual: the gate accepts failure while presenting itself as a positive decode oracle.

## Fix

Choose and pin one contract. If the recovery inter must decode, replace the catch-all error arm with an `Ok` assertion and verify output dimensions or pixels. If entropy-bank poisoning intentionally permits failure, rename the test and narrow the public documentation, then add a separate positive oracle for whatever reuse guarantee remains. Estimated effort: small.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `f274787175bf93421861efe348861a3646659cec` (test(dec): require inter recovery to decode).
The original reference `267e292b` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `3fc008f7946da9f0dc3335c6d086962fbfbefcd4` has this fixing commit as its immediate parent.
mid_tile_failure_then_valid_inter_decodes replaces permissive Err(_) match arm with decode_frame(...).expect("valid inter after a failed inter decodes") and width/height assertions.
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

## Notes

Confirmed by the test-coverage lens, the lead's direct documentation/test comparison, and the standing devil's-advocate pass. No test was run in this static review pass.
