# RVP9-BUG-KIL-00114 — Throughput ratchet selects baselines from the wrong build profile

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/throughput ratchet
- **Raised:** 2026-08-22T19:31:41Z
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
- **State history:** Open (2026-08-22T19:31:41Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T03:14:11Z, deltic:auto role=fix run=fix-20261016T030427Z-c583c67f branch=task/bug-RVP9-BUG-KIL-00114-run-fix-20261016T030427Z-c583c67f code=34a232d6ba4ab4cc00b7a1f99181f258259e916f gate=manual) -> Closed (2026-10-18T18:07:44Z, deltic:manual role=verify run=verify-20261018T180101Z-a2338d05 verified_fix_run=fix-20261016T030427Z-c583c67f verdict=close)

## Observation

Throughput CSV rows store build_profile_hash specifically to disambiguate debug and release runs at crates/rvp9-bench/src/throughput.rs:419-430. latest_per_fixture at lines 557-594 filters only pinned host membership and keys its reduction by fixture, so the newest row wins regardless of build profile. decode_throughput_ratchet.rs:130-173 consumes that result without a profile filter. Expected: current measurements compare with the same host and build identity. Actual: a newer incompatible profile can replace the intended baseline and create a false pass or false regression. Select by fixture, exact host, and explicit build identity, and add mixed-host and mixed-profile regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T18:07:44Z, deltic:auto run=verify-20261018T180101Z-a2338d05 verified_fix_run=fix-20261016T030427Z-c583c67f verdict=close)

Verifier note: Manual verify: Both retained mixed-identity assertions pass before/after exact restore. Removing only the production build-profile filter makes the selector choose timestamp 80 vs 20 and the ratchet choose 30 vs 10. Exact restore passes. The stale bug-record Fix placeholder remains unchanged; two focused tests only.

## Notes
