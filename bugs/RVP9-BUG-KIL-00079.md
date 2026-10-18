# RVP9-BUG-KIL-00079 — ScaleFactors accepts zero dimensions then divides by zero

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** core/inter scaling
- **Raised:** 2026-08-22T08:57:01Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T161001Z-d8fb7c20
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00079-run-verify-20261018T161001Z-d8fb7c20
- **Owner base:** 4d1c3c8108d7f2ea9a2e03329602fd019846127e
- **Owner fingerprint:** sha256:370829cb6fdc071ba7b71dc5a20cd75b8e2e7a3bac5adbaab5a8964f47201c51
- **Owner since:** 2026-10-18T16:10:01Z
- **Owner until:** 2026-10-18T20:10:01Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T08:57:01Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T05:47:54Z, deltic:auto role=fix run=fix-20261016T054106Z-5067f680 branch=task/bug-RVP9-BUG-KIL-00079-run-fix-20261016T054106Z-5067f680 code=c49db873520cbded462e47a99e7e1182bdf7b5cc gate=manual)

## Observation

`ScaleFactors::new` documents that dimension pairs failing `valid_ref_frame_size` produce `REF_INVALID_SCALE` at `crates/rvp9-core/src/inter/scale.rs:82-100`. However, `valid_ref_frame_size` at `:155-168` accepts a zero reference/current pair because all four inequalities hold for zero. The constructor then calls `fixed_point_scale_factor` at `:102-103`, whose debug-only assertion at `:176` is followed by integer division by the zero current dimension at `:181`.

Expected: zero dimensions are rejected as invalid geometry and produce the same invalid-scale sentinel as other invalid pairs. Actual: `ScaleFactors::new(0, 0, 0, 0)` panics in every build profile; mixed zero-axis pairs can reach the same failure. Valid VP9 frame headers exclude zero dimensions, so the confirmed impact is the public safe API rather than a malformed-bitstream decoder path.

## Proposed resolution

Reject any zero reference or current axis in `valid_ref_frame_size` before the ratio inequalities. Add focused cases for each zero axis and the all-zero tuple, asserting `!valid_ref_frame_size`, `!ScaleFactors::is_valid()`, and zero step fields. Estimated effort: trivial.

## Review evidence

Confirmed independently by the correctness reviewer, the lead's call trace, and the standing devil's advocate. No tests or application runs were performed in this review pass.

## Fix

<unfixed — raised only>

## Notes
