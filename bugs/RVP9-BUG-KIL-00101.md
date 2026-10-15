# RVP9-BUG-KIL-00101 — Q-mode lossless request is silently encoded at qindex 1

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/rate-control integration
- **Raised:** 2026-08-22T18:32:48Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261015T233958Z-f4ab4ecc
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00101-run-fix-20261015T233958Z-f4ab4ecc
- **Owner base:** 8263d139e4a967e55eb40b74b97e345fb006dacb
- **Owner fingerprint:** -
- **Owner since:** 2026-10-15T23:39:58Z
- **Owner until:** 2026-10-16T03:39:58Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T18:32:48Z, raised via `deltic bugs new`)

## Observation

`RateControlConfig::cq_level` says Q mode may select qindex 0 as the lossless edge at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\rate_control\config.rs:117`. The Q-mode controller returns that value unchanged at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\rate_control\state.rs:532`, but the streaming frame loop silently replaces every selected qindex 0 with 1 at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\frame_loop.rs:2983` because its simulator does not support lossless coding.

A caller can therefore construct the documented Q-mode configuration with `cq_level = 0`, receive a successful encoded frame, and observe `EncodedFrame::q_index == 1` rather than the requested lossless qindex. The same clamp can override a CBR pick when callers explicitly permit qindex 0. Expected: unsupported lossless configurations fail during `StreamEncoder` construction, or qindex 0 is encoded losslessly. Actual: configuration succeeds and its requested/picked quantizer is changed without an error.

## Fix

Until the full qindex-0 transform, quantizer, and loop-filter path exists, reject configurations that can require qindex 0 instead of clamping after rate control. If lossless support is implemented, carry the lossless flag through reconstruction and packet writing and remove the clamp. Add end-to-end Q-mode qindex-0 coverage that checks either construction rejection or the emitted frame's reported and wire quantizer. Estimated effort: small for explicit rejection; large for lossless support.

## Notes

Confirmed by the correctness lens, the test-coverage lens, the lead's source trace, and the devil's-advocate pass. Existing Q-mode tests use nonzero values; no code or test was run during this static review.
