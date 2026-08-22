# RVP9-BUG-KIL-00114 — Throughput ratchet selects baselines from the wrong build profile

- **State:** Open
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
- **State history:** Open (2026-08-22T19:31:41Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Throughput CSV rows store build_profile_hash specifically to disambiguate debug and release runs at crates/rvp9-bench/src/throughput.rs:419-430. latest_per_fixture at lines 557-594 filters only pinned host membership and keys its reduction by fixture, so the newest row wins regardless of build profile. decode_throughput_ratchet.rs:130-173 consumes that result without a profile filter. Expected: current measurements compare with the same host and build identity. Actual: a newer incompatible profile can replace the intended baseline and create a false pass or false regression. Select by fixture, exact host, and explicit build identity, and add mixed-host and mixed-profile regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
