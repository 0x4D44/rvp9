# RVP9-BUG-KIL-00087 — PSNR delta summary returns non-finite statistics

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** harness/bdrate
- **Raised:** 2026-08-22T13:20:52Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T165858Z-f3876347
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00087-run-verify-20261018T165858Z-f3876347
- **Owner base:** e3ca07aa934274ab2a10e0621c46c5221ccf5ec4
- **Owner fingerprint:** sha256:8a6ce181e5c4d814932a3aa3286b48b2fd80f6df55abe1b184e8d22f33c44fea
- **Owner since:** 2026-10-18T16:58:58Z
- **Owner until:** 2026-10-18T20:58:58Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T13:20:52Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T07:59:11Z, deltic:auto role=fix run=fix-20261016T074641Z-b354a429 branch=task/bug-RVP9-BUG-KIL-00087-run-fix-20261016T074641Z-b354a429 code=e61bd7e2be6e2971edf589a932678589ae31ac93 gate=manual)

## Observation

`psnr_at_bitrate` rejects non-finite samples in the reference curve at
`crates/rvp9-harness/src/bdrate.rs:131-139`, but `psnr_delta_summary` does not
validate the rvp9 sample's `psnr_r` before subtracting it at lines 221-224. A
covered rvp9 point containing `NaN` or infinity is pushed into `deltas`, and
the function returns `Some(PsnrDeltaSummary)` with non-finite mean/min/max at
lines 227-241.

Expected: invalid samples produce no quality signal. Actual: the API returns a
present summary whose statistics cannot be meaningfully compared or reported.

## Fix

<unfixed — raised only>

## Notes

Validate the complete rvp9 curve before interpolation and return `None` for any
non-positive/non-finite bitrate or non-finite PSNR, matching the reference-curve
policy. Add `NaN` and infinity regressions. Confirmed statically; no tests ran.
