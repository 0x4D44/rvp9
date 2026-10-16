# RVP9-BUG-KIL-00111 — Realtime benchmark runs collide through one default IVF path

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/realtime
- **Raised:** 2026-08-22T19:31:16Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T040432Z-f59e60cf
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00111-run-fix-20261016T040432Z-f59e60cf
- **Owner base:** 2a10f424c73c4facfda8431e87c4de61ab639a27
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T04:04:32Z
- **Owner until:** 2026-10-16T08:04:32Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:31:16Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

crates/rvp9-bench/src/bin/realtime.rs:118-120 assigns every invocation the same system-temp output path, rvp9_realtime_bench.ivf. The external encoder writes that path at lines 379-389 and the benchmark later counts it at lines 391-399. Concurrent runs can overwrite or count each other output, and a stale file can survive an encoder that does not replace it correctly. Expected: default runs have isolated artifacts and measure only their own output. Actual: process identity is absent from the path. Use a scoped unique temporary directory with create-new semantics, or require an explicit output path and reject pre-existing artifacts. Add concurrent-run and stale-output regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
