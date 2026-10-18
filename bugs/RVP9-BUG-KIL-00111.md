# RVP9-BUG-KIL-00111 — Realtime benchmark runs collide through one default IVF path

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/realtime
- **Raised:** 2026-08-22T19:31:16Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T162039Z-53141d13
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00111-run-verify-20261018T162039Z-53141d13
- **Owner base:** 6d4a10437b0c32c8d981d2603cda5a1fbd71b3b9
- **Owner fingerprint:** sha256:1147996b405d9c49945637531a14e72af1aacb55894f04fc3430302cb0626244
- **Owner since:** 2026-10-18T16:20:39Z
- **Owner until:** 2026-10-18T20:20:39Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:31:16Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T04:20:37Z, deltic:auto role=fix run=fix-20261016T040432Z-f59e60cf branch=task/bug-RVP9-BUG-KIL-00111-run-fix-20261016T040432Z-f59e60cf code=07b1a8e5689358e0d095452b6bd71a2726220c0f gate=manual)

## Observation

crates/rvp9-bench/src/bin/realtime.rs:118-120 assigns every invocation the same system-temp output path, rvp9_realtime_bench.ivf. The external encoder writes that path at lines 379-389 and the benchmark later counts it at lines 391-399. Concurrent runs can overwrite or count each other output, and a stale file can survive an encoder that does not replace it correctly. Expected: default runs have isolated artifacts and measure only their own output. Actual: process identity is absent from the path. Use a scoped unique temporary directory with create-new semantics, or require an explicit output path and reject pre-existing artifacts. Add concurrent-run and stale-output regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
