# RVP9-BUG-KIL-00101 — Q-mode lossless request is silently encoded at qindex 1

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/rate-control integration
- **Raised:** 2026-08-22T18:32:48Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T052709Z-e6646641
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00101-run-verify-20261019T052709Z-e6646641
- **Owner base:** 6f220754d6f82a104ba3822fb01319b7295ad505
- **Owner fingerprint:** sha256:e65c31a088297c212ab25ef070edfc3a7e6333bcf76c50dfdfa60335058bc775
- **Owner since:** 2026-10-19T05:27:09Z
- **Owner until:** 2026-10-19T09:27:09Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T18:32:48Z, raised via `deltic bugs new`) -> Fixed (2026-10-15T23:49:43Z, deltic:auto role=fix run=fix-20261015T233958Z-f4ab4ecc branch=task/bug-RVP9-BUG-KIL-00101-run-fix-20261015T233958Z-f4ab4ecc code=b6d3afc00423438c5026cdd0cb812acf5e4f8a97 gate=manual)

## Observation

`RateControlConfig::cq_level` says Q mode may select qindex 0 as the lossless edge at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\rate_control\config.rs:117`. The Q-mode controller returns that value unchanged at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\rate_control\state.rs:532`, but the streaming frame loop silently replaces every selected qindex 0 with 1 at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\frame_loop.rs:2983` because its simulator does not support lossless coding.

A caller can therefore construct the documented Q-mode configuration with `cq_level = 0`, receive a successful encoded frame, and observe `EncodedFrame::q_index == 1` rather than the requested lossless qindex. The same clamp can override a CBR pick when callers explicitly permit qindex 0. Expected: unsupported lossless configurations fail during `StreamEncoder` construction, or qindex 0 is encoded losslessly. Actual: configuration succeeds and its requested/picked quantizer is changed without an error.

## Fix

Until the full qindex-0 transform, quantizer, and loop-filter path exists, reject configurations that can require qindex 0 instead of clamping after rate control. If lossless support is implemented, carry the lossless flag through reconstruction and packet writing and remove the clamp. Add end-to-end Q-mode qindex-0 coverage that checks either construction rejection or the emitted frame's reported and wire quantizer. Estimated effort: small for explicit rejection; large for lossless support.

## Notes

Confirmed by the correctness lens, the test-coverage lens, the lead's source trace, and the devil's-advocate pass. Existing Q-mode tests use nonzero values; no code or test was run during this static review.
