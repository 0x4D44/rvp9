# RVP9-BUG-FLU-00056 — AArch64 IDCT16 destination span check can wrap before unsafe stores

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** core/transforms/inverse-neon
- **Raised:** 2026-10-18T16:12:39Z
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
- **State history:** Open (2026-10-18T16:12:39Z, raised via `deltic bugs new --land`) -> Fixed (2026-10-18T16:44:38Z, deltic:auto role=fix run=fix-20261018T163646Z-66a5c8ca branch=task/bug-RVP9-BUG-FLU-00056-run-fix-20261018T163646Z-66a5c8ca code=352e2be84ef18b5e5530349ffcec932a9e71430c gate=manual) -> Closed (2026-10-18T17:10:38Z, deltic:manual role=verify run=verify-20261018T170008Z-73d30c0f verified_fix_run=fix-20261018T163646Z-66a5c8ca verdict=close)

## Observation

During the Open-queue fix for RVP9-BUG-FLU-00043, source review found that the safe public inverse IDCT16 entry points reach transforms/inverse_neon.rs::idct16x16_add on native AArch64. That helper checks assert!(stride >= 16) and assert!(dest.len() >= stride * 15 + 16), then its raw-pointer kernel computes dest.add(row * stride + column + lane) for rows through 15.

With release overflow checks disabled on a 64-bit target, stride = usize::MAX / 15 + 1 makes stride * 15 + 16 wrap to 30. A destination slice of length 30 therefore passes both assertions, although the second row begins at offset 1229782938247303442. The raw pointer formed for row 1 lies outside the supplied allocation. A normal safe call to public idct16x16_256_add with input [0;256], destination [128;30] and that stride can therefore reach out-of-allocation pointer arithmetic/writes on the normal native NEON route. The sparse 10/38 entry points share the same helper.

Expected: malformed destination geometry is rejected with checked arithmetic before any unsafe pointer calculation. Actual: release-mode arithmetic in the bounds assertion can wrap and accept invalid geometry. This is source-derived; the invalid unsafe call was deliberately not executed. The scalar-mask routing fix does not address this native bounds defect. This is separate from RVP9-BUG-FLU-00049 (x86 forward DCT) and RVP9-BUG-FLU-00046 (AArch64 convolution strides). No code fix is included in this record.

Evidence fingerprint: `manual:v1:aarch64-idct16-span-wrap`


## Fix

The required 16×16 destination span now uses checked multiplication and addition, and an unconditional shared guard validates the stride and slice length before scalar writes or entry into the unsafe AArch64 NEON kernel. The same guard covers the DC-only scalar helper. The regression checks both multiplication and addition overflow and verifies rejection without invoking an unsafe kernel.

### Verification summary (2026-10-18T17:10:38Z, deltic:auto run=verify-20261018T170008Z-73d30c0f verified_fix_run=fix-20261018T163646Z-66a5c8ca verdict=close)

Verifier note: Manual verify: Independent native arm64 verification on the fixed build. The retained overflow test passes in optimized dev with CARGO_PROFILE_DEV_OVERFLOW_CHECKS=false; the workspace release fat-LTO test build exceeded 300s before execution. Replacing only checked span arithmetic with the original unchecked multiply/add makes the retained test fail: the overflowing stride produces Some(30) instead of None. Restored rerun passes and inverse.rs hash matches its pre-mutation SHA-256; scoped diff is empty. The unsafe public call with invalid geometry was deliberately not executed; source routing ...

## Notes
