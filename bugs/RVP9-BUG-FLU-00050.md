# RVP9-BUG-FLU-00050 — 32x32 SIMD IDCT forms an out-of-allocation pointer after the final row

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** core/dsp x86 IDCT
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
- **State history:** Open (2026-10-17T20:46:57Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T17:44:06Z, deltic:auto role=fix run=fix-20261018T172937Z-b60b783e branch=task/bug-RVP9-BUG-FLU-00050-run-fix-20261018T172937Z-b60b783e code=925594df719a4d12005bb7af7cfe6783d156b2cc gate=manual)

## Observation

On an x86_64 host using the SIMD table, a safe call through SimdDispatch::idct32x32_add with a valid 1024-byte destination and stride 32 reaches the SSE2 or AVX2 32x32 IDCT storage helper. In store_buffer_8x32 at crates/rvp9-core/src/dsp/x86_64/idct32x32_sse2_1024.rs:215-233 and store_buffer_16x32 at crates/rvp9-core/src/dsp/x86_64/idct32x32_avx2_1024.rs:171-184, each loop unconditionally advances dst by stride after storing row 31. The final SSE2 strip starts at offset 24 and forms offset 1048; the AVX2 strip starts at offset 16 and forms offset 1040. Both exceed one-past the 1024-byte allocation, although every requested pixel store fits the valid tight 32x32 block. Rust pointer add requires the result to remain in the allocation or one-past. Expected: valid transform geometry performs only in-allocation pointer arithmetic. Actual: these safe dispatch paths form out-of-allocation pointers after their final row. This is source-derived; no UB detector was run.

## Fix

The SSE2 8x32 and AVX2 16x32 final-store loops now advance the destination pointer only when another row remains. They share a guarded row-advance helper, so the final store does not form a pointer beyond the tight destination allocation. Added an exact-final-row pointer regression: under Miri it failed before the guard because the pointer advanced by one stride, then passed after the fix. The x86_64 `idct32x32` filter passed all 70 tests under Rosetta; scoped x86 Clippy, rustfmt, `git diff --check`, and the exact smoke passed.

## Notes
