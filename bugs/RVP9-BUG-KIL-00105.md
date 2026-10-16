# RVP9-BUG-KIL-00105 — E12 diagnostic leaf API panics on unchecked geometry

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** encoder/e12 diagnostic API
- **Raised:** 2026-08-22T18:32:52Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T080335Z-5acb068e
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00105-run-fix-20261016T080335Z-5acb068e
- **Owner base:** fd09a17f9a99317552f5f00ddc2ac02922a24bc0
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T08:03:35Z
- **Owner until:** 2026-10-16T12:03:35Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T18:32:52Z, raised via `deltic bugs new`)

## Observation

The public diagnostic entry point `e12stat::record_leaf` accepts arbitrary slice, stride, offset, and block geometry at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\e12stat.rs:251`. When `RVP9_E12_ORACLE` enables a sampled call, its private `pred_sse` helper uses a fixed 64 by 64 prediction buffer at `e12stat.rs:152`, passes caller-controlled `bw` and `bh` to the predictor at `:153`, and later indexes caller-controlled source row windows at `:168`. It validates only the reference prediction window, not the maximum block size or source coverage.

Expected: this safe public API rejects or safely ignores invalid geometry. Actual: a sampled enabled call with a block larger than 64 by 64 or insufficient source coverage can panic through predictor or slice indexing. Normal encoder callers use VP9-sized blocks, so the confirmed impact is the public diagnostic API rather than the ordinary encode path.

## Fix

Validate nonzero `bw`/`bh <= 64`, checked source and reference row coverage, strides, offsets, and integer conversions before prediction. Return a `Result` or make invalid samples a documented no-op. Add enabled-oracle regressions for oversized blocks, short slices, and overflowing geometry. Estimated effort: small.

## Notes

Confirmed by the security lens, the lead's source trace, and the devil's-advocate pass. No code or test was run during this static review.
