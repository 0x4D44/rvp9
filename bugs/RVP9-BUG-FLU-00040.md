# RVP9-BUG-FLU-00040 — Tile profiling feature fails to compile after scope timestamp binding was lost

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** decoder/profile-tiles
- **Raised:** 2026-10-16T01:16:30Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T171227Z-f6b0f3ce
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00040-run-verify-20261018T171227Z-f6b0f3ce
- **Owner base:** 07ddbf5d0120b2b544a23278e03064e36963cbcb
- **Owner fingerprint:** sha256:7b47d21432514c0f5abd60ff618e047886e9935ab52de0a855cc1c047f250b84
- **Owner since:** 2026-10-18T17:12:27Z
- **Owner until:** 2026-10-18T21:12:27Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-16T01:16:30Z, raised via `deltic bugs new --land` model=gpt-6.1-sol@medium) -> Fixed (2026-10-16T01:18:45Z, deltic:auto role=fix run=fix-20261016T011650Z-96ec5bbd branch=task/bug-RVP9-BUG-FLU-00040-run-fix-20261016T011650Z-96ec5bbd code=2d327e4e5ec5ddecffd8d8bcffcb37730893f35d gate=manual)

## Observation

Required x86_64 profile-tiles validation for held RVP9-BUG-KIL-00110 fails with E0425 at both keyframe and inter-frame timing snapshots: _submit_done_cycles is referenced but never bound. Reproduced twice in the held worktree with unchanged upstream scope code; origin/main contains the same lost bindings. The scope closure still returns its last-submit timestamp. Expected: cargo check -p rvp9-decoder --target x86_64-apple-darwin --features profile-tiles succeeds. Actual: four missing-variable errors. Separate narrow incidental unblocker: restore the return-value binding at the two scope calls, with no instrumentation or codec behavior redesign.

Evidence fingerprint: `trunk-red:v2:rvp9-tile-profiling-feature-fails-to-compile-af-d2129c0744ce3584`


## Fix

<unfixed — raised only>

## Notes
