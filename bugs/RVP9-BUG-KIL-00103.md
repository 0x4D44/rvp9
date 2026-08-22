# RVP9-BUG-KIL-00103 — StreamEncoder accepts unsupported speed values

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/config validation
- **Raised:** 2026-08-22T18:32:50Z
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
- **State history:** Open (2026-08-22T18:32:50Z, raised via `deltic bugs new`)

## Observation

`EncoderConfig::speed` documents the accepted VP9 range as `-9..=9` at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\frame_loop.rs:497`. `StreamEncoder::new_with_parallelism_and_buffer_geometry` validates dimensions, bit depth, GOP, layers, framerate, bitrate, and rate control around `frame_loop.rs:2110`, but never validates speed.

Unsupported values therefore construct successfully. The dispatcher takes `speed.unsigned_abs()` at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-core\src\mode_decision\speed_features.rs:198`; threshold and equality checks then create an undocumented mixture of speed-9-or-faster and non-speed-8 behavior. Expected: values outside the documented range return `EncodeError::InvalidConfig`. Actual: they are silently accepted and alter coding policy.

## Fix

Validate `-9..=9` before constructing any encoder state and return `InvalidConfig` outside that range. Add boundary regressions for -9, 9, -10, and 10 through the public constructor. Estimated effort: small.

## Notes

Confirmed by the maintainability/API lens, the lead's source trace, and the devil's-advocate pass. No matching open bug was found. No code or test was run during this static review.
