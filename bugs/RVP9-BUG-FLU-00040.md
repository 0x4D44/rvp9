# RVP9-BUG-FLU-00040 — Tile profiling feature fails to compile after scope timestamp binding was lost

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** decoder/profile-tiles
- **Raised:** 2026-10-16T01:16:30Z
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
- **State history:** Open (2026-10-16T01:16:30Z, raised via `deltic bugs new --land` model=gpt-6.1-sol@medium) -> Fixed (2026-10-16T01:18:45Z, deltic:auto role=fix run=fix-20261016T011650Z-96ec5bbd branch=task/bug-RVP9-BUG-FLU-00040-run-fix-20261016T011650Z-96ec5bbd code=2d327e4e5ec5ddecffd8d8bcffcb37730893f35d gate=manual) -> Closed (2026-10-18T17:14:37Z, deltic:manual role=verify run=verify-20261018T171227Z-f6b0f3ce verified_fix_run=fix-20261016T011650Z-96ec5bbd verdict=close)

## Observation

Required x86_64 profile-tiles validation for held RVP9-BUG-KIL-00110 fails with E0425 at both keyframe and inter-frame timing snapshots: _submit_done_cycles is referenced but never bound. Reproduced twice in the held worktree with unchanged upstream scope code; origin/main contains the same lost bindings. The scope closure still returns its last-submit timestamp. Expected: cargo check -p rvp9-decoder --target x86_64-apple-darwin --features profile-tiles succeeds. Actual: four missing-variable errors. Separate narrow incidental unblocker: restore the return-value binding at the two scope calls, with no instrumentation or codec behavior redesign.

Evidence fingerprint: `trunk-red:v2:rvp9-tile-profiling-feature-fails-to-compile-af-d2129c0744ce3584`


## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T17:14:37Z, deltic:auto run=verify-20261018T171227Z-f6b0f3ce verified_fix_run=fix-20261016T011650Z-96ec5bbd verdict=close)

Verifier note: Manual verify: Independent x86_64 profile-tiles build verification: the recorded fix 2d327e4e is present; baseline and restored cargo checks pass. Removing only the two returned timestamp bindings reproduces all four E0425 errors at the keyframe/inter-frame submit and join snapshots. Exact source restored and scoped diff clean. Cross-compilation only; no runtime profiling test. Existing apple-m4/x87 target warnings also occur on the passing baseline and are unrelated.

## Notes
