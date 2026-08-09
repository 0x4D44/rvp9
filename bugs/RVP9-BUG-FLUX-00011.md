# RVP9-BUG-FLUX-00011 — s11 intra restriction test rejects a valid control mode

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** core/mode-decision tests
- **Raised:** 2026-08-09T20:08:59Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260809T224037Z-p62662-n028162000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00011-run-fix-20260809T224037Z-p62662-n028162000-c1
- **Owner base:** 1903a7dd691bea74b60f078262eacb9ef0e4aa0e
- **Owner fingerprint:** -
- **Owner since:** 2026-08-09T22:40:37Z
- **Owner until:** 2026-08-10T00:40:37Z
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
