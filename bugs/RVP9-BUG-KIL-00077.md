# RVP9-BUG-KIL-00077 — Mode-decision 16x16 and 32x32 IDCT bypass SIMD dispatch

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/mode-decision transforms
- **Raised:** 2026-08-22T08:56:35Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T070249Z-6db58ac6
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00077-run-verify-20261019T070249Z-6db58ac6
- **Owner base:** 754088c5a2d900770552051fbbbf4ee3b09e1d18
- **Owner fingerprint:** sha256:bfdf8b144151c188723b8714a1e40133e099f72a2b71e5acddb8ac5ebf2a63ff
- **Owner since:** 2026-10-19T07:02:49Z
- **Owner until:** 2026-10-19T11:02:49Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T08:56:35Z, raised via `deltic bugs new`) -> Fixed (2026-08-22T13:20:45Z, deltic:auto role=fix run=fix-20260822T131414Z-6ea9886c branch=task/bug-RVP9-BUG-KIL-00077-run-fix-20260822T131414Z-6ea9886c code=daae16aa gate=manual)

## Observation

Static source review found that `reconstruct_via_round_trip` obtains the runtime DSP table at `crates/rvp9-core/src/mode_decision/inter.rs:5419` and uses `dsp.idct8x8_add()` for 8x8 candidates at `:5811-5818`, but directly calls scalar `idct16x16_256_add` and `idct32x32_1024_add` for 16x16 and 32x32 candidates at `:5820-5833`. Runtime-dispatched SIMD slots for both larger transforms exist at `crates/rvp9-core/src/dsp/dispatch.rs:1431-1469` and are exposed at `:2800-2817`.

Expected: this dense per-candidate reconstruction path uses the runtime dispatch table for every inverse-transform size with an accelerated slot. Actual: 16x16 and 32x32 candidates always execute the scalar kernels, leaving the available SSE2/AVX2 implementations idle in a hot encoder search path. The static call graph proves the bypass; this review did not run a benchmark, so the exact throughput loss is not measured.

## Proposed resolution

Call `dsp.idct16x16_add()` and `dsp.idct32x32_add()` in the corresponding match arms and remove the direct scalar imports. Add a dispatch-routing regression that distinguishes scalar from accelerated slots without timing, then benchmark a representative large-transform encode to quantify the recovered throughput. Estimated effort: small.

## Review evidence

Confirmed independently by the performance reviewer, the lead's source trace, and the standing devil's advocate. No tests, benchmarks, or application runs were performed in this review pass.

## Fix

<unfixed — raised only>

## Notes
