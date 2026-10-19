# RVP9-BUG-KIL-00077 — Mode-decision 16x16 and 32x32 IDCT bypass SIMD dispatch

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/mode-decision transforms
- **Raised:** 2026-08-22T08:56:35Z
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
- **State history:** Open (2026-08-22T08:56:35Z, raised via `deltic bugs new`) -> Fixed (2026-08-22T13:20:45Z, deltic:auto role=fix run=fix-20260822T131414Z-6ea9886c branch=task/bug-RVP9-BUG-KIL-00077-run-fix-20260822T131414Z-6ea9886c code=daae16aa gate=manual) -> Closed (2026-10-19T07:07:10Z, deltic:manual role=verify run=verify-20261019T070249Z-6db58ac6 verified_fix_run=fix-20260822T131414Z-6ea9886c verdict=close)

## Observation

Static source review found that `reconstruct_via_round_trip` obtains the runtime DSP table at `crates/rvp9-core/src/mode_decision/inter.rs:5419` and uses `dsp.idct8x8_add()` for 8x8 candidates at `:5811-5818`, but directly calls scalar `idct16x16_256_add` and `idct32x32_1024_add` for 16x16 and 32x32 candidates at `:5820-5833`. Runtime-dispatched SIMD slots for both larger transforms exist at `crates/rvp9-core/src/dsp/dispatch.rs:1431-1469` and are exposed at `:2800-2817`.

Expected: this dense per-candidate reconstruction path uses the runtime dispatch table for every inverse-transform size with an accelerated slot. Actual: 16x16 and 32x32 candidates always execute the scalar kernels, leaving the available SSE2/AVX2 implementations idle in a hot encoder search path. The static call graph proves the bypass; this review did not run a benchmark, so the exact throughput loss is not measured.

## Proposed resolution

Call `dsp.idct16x16_add()` and `dsp.idct32x32_add()` in the corresponding match arms and remove the direct scalar imports. Add a dispatch-routing regression that distinguishes scalar from accelerated slots without timing, then benchmark a representative large-transform encode to quantify the recovered throughput. Estimated effort: small.

## Review evidence

Confirmed independently by the performance reviewer, the lead's source trace, and the standing devil's advocate. No tests, benchmarks, or application runs were performed in this review pass.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-19T07:07:10Z, deltic:auto run=verify-20261019T070249Z-6db58ac6 verified_fix_run=fix-20260822T131414Z-6ea9886c verdict=close)

Verifier note: Manual verify: On origin/main, commit 4d7305951d0a44bdce9550a7eef049542f1ba834 is the landed IDCT dispatch fix and is an ancestor of trunk; reconstruct_via_round_trip routes Tx8x8, Tx16x16, and Tx32x32 through RoundTripIdctDispatch populated from the runtime DSP table. The recorded fix hash daae16aa does not resolve in this checkout; the matching landed change is 4d730595. Focused cargo test -p rvp9-core round_trip_idct_dispatch_routes_each_realtime_tx_size passed (1 selected). Reversing Tx16x16 to scalar made its assertion fail (left 8, expected 16); reversing Tx32x32 to scalar failed (left 1...

## Notes
