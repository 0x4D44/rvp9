# RVP9-BUG-KIL-00105 — E12 diagnostic leaf API panics on unchecked geometry

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** encoder/e12 diagnostic API
- **Raised:** 2026-08-22T18:32:52Z
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
- **State history:** Open (2026-08-22T18:32:52Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T08:19:55Z, deltic:auto role=fix run=fix-20261016T080335Z-5acb068e branch=task/bug-RVP9-BUG-KIL-00105-run-fix-20261016T080335Z-5acb068e code=2f2319a08e5da43791628062360ce85df9c98b06 gate=manual) -> Closed (2026-10-18T17:48:23Z, deltic:manual role=verify run=verify-20261018T174232Z-b6fcf8f9 verified_fix_run=fix-20261016T080335Z-5acb068e verdict=close)

## Observation

The public diagnostic entry point `e12stat::record_leaf` accepts arbitrary slice, stride, offset, and block geometry at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\e12stat.rs:251`. When `RVP9_E12_ORACLE` enables a sampled call, its private `pred_sse` helper uses a fixed 64 by 64 prediction buffer at `e12stat.rs:152`, passes caller-controlled `bw` and `bh` to the predictor at `:153`, and later indexes caller-controlled source row windows at `:168`. It validates only the reference prediction window, not the maximum block size or source coverage.

Expected: this safe public API rejects or safely ignores invalid geometry. Actual: a sampled enabled call with a block larger than 64 by 64 or insufficient source coverage can panic through predictor or slice indexing. Normal encoder callers use VP9-sized blocks, so the confirmed impact is the public diagnostic API rather than the ordinary encode path.

## Fix

Validate nonzero `bw`/`bh <= 64`, checked source and reference row coverage, strides, offsets, and integer conversions before prediction. Return a `Result` or make invalid samples a documented no-op. Add enabled-oracle regressions for oversized blocks, short slices, and overflowing geometry. Estimated effort: small.

### Verification summary (2026-10-18T17:48:23Z, deltic:auto run=verify-20261018T174232Z-b6fcf8f9 verified_fix_run=fix-20261016T080335Z-5acb068e verdict=close)

Verifier note: Manual verify: Both retained E12 tests passed before and after exact restore. Reversing source-stride validation made the invalid-geometry no-op assertion fail: both short-stride cases accumulated samples without panic; reads remained within the backing slice. Focused tests only.

## Notes

Confirmed by the security lens, the lead's source trace, and the devil's-advocate pass. No code or test was run during this static review.
