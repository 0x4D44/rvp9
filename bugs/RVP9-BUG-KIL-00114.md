# RVP9-BUG-KIL-00114 — Throughput ratchet selects baselines from the wrong build profile

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/throughput ratchet
- **Raised:** 2026-08-22T19:31:41Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T180101Z-a2338d05
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00114-run-verify-20261018T180101Z-a2338d05
- **Owner base:** 8d2d2dec5084588149d18454423d84c34ef4e368
- **Owner fingerprint:** sha256:dd4b42bb761e2d748329822ea5fc201313eb51aa26c14bc2fb981dd99ff8e5fb
- **Owner since:** 2026-10-18T18:01:01Z
- **Owner until:** 2026-10-18T22:01:01Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:31:41Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T03:14:11Z, deltic:auto role=fix run=fix-20261016T030427Z-c583c67f branch=task/bug-RVP9-BUG-KIL-00114-run-fix-20261016T030427Z-c583c67f code=34a232d6ba4ab4cc00b7a1f99181f258259e916f gate=manual)

## Observation

Throughput CSV rows store build_profile_hash specifically to disambiguate debug and release runs at crates/rvp9-bench/src/throughput.rs:419-430. latest_per_fixture at lines 557-594 filters only pinned host membership and keys its reduction by fixture, so the newest row wins regardless of build profile. decode_throughput_ratchet.rs:130-173 consumes that result without a profile filter. Expected: current measurements compare with the same host and build identity. Actual: a newer incompatible profile can replace the intended baseline and create a false pass or false regression. Select by fixture, exact host, and explicit build identity, and add mixed-host and mixed-profile regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
