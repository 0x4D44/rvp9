# RVP9-BUG-FLUX-00011 — s11 intra restriction test rejects a valid control mode

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** core/mode-decision tests
- **Raised:** 2026-08-09T20:08:59Z
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
- **State history:** Open (2026-08-09T20:08:59Z, raised via `deltic bugs new`)

## Observation

Observation: `cargo test -p rvp9-core --lib mode_decision::inter::tests::s11_rt_intra_dc_only_low_content_restricts_to_dc -- --exact` fails deterministically at the control assertion because `picked_intra_mode` can be a mode outside DC/V/H. The same failure reproduces from a clean archive of commit f3a330c2, so it is not caused by the denoiser worktree. Expected: the focused test and full core suite pass while still proving that the active gate restricts normal content to DC; actual: the control-side anti-stub assertion rejects a valid broader-mode result.

## Fix

<unfixed — raised only>

## Notes
