# RVP9-BUG-FLU-00049 — SSE2 forward DCT span check can wrap before unsafe input loads

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** core/transforms x86 SSE2
- **Raised:** 2026-10-17T20:46:57Z
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
- **State history:** Open (2026-10-17T20:46:57Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T17:29:17Z, deltic:auto role=fix run=fix-20261018T171040Z-454dd919 branch=task/bug-RVP9-BUG-FLU-00049-run-fix-20261018T171040Z-454dd919 code=48b442451cc86416657ac75ca5e63424274e5fb9 gate=manual)

## Observation

On 64-bit x86 release builds, the safe public rvp9_core::transforms::fdct8x8 wrapper routes to SSE2, whose validation at crates/rvp9-core/src/transforms/forward_sse2.rs:149-158 checks input.len() >= 7 * stride + 8 using unchecked arithmetic. With a 64-element input slice and stride = usize::MAX / 7, the required-span expression wraps to 6, so the assertion passes; the unsafe SSE2 kernel then forms row pointers beyond the input allocation. Expected: overflowing geometry is rejected before raw pointer operations. Actual: the safe public API accepts the wrapped span and reaches out-of-allocation pointer arithmetic in release. Source review also found sibling unchecked span guards in forward 16x16, Hadamard, minmax, VBP, integer projection, SSE block, and related x86 wrappers; this record anchors the concrete public 8x8 DCT path. This is distinct from RVP9-BUG-FLU-00032, whose fixed scope is square subtraction. No runtime reproduction was attempted.

## Fix

The SSE2 8x8 DCT wrapper now computes the required input span with checked multiplication and addition before entering the unsafe kernel. Overflow is rejected instead of allowing `7 * stride + 8` to wrap. Added a regression covering the valid stride-8 span and both multiplication- and addition-overflow boundaries. The overflow-disabled regression failed before the fix with `Some(6)` for `stride = usize::MAX / 7`, then passed after the fix. On x86_64 under Rosetta, all eight focused `fdct8x8` tests passed; scoped x86 Clippy, rustfmt, and `git diff --check` passed.

## Notes
