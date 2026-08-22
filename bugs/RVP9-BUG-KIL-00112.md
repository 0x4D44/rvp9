# RVP9-BUG-KIL-00112 — Benchmark tools wait forever for external encoders and decoders

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/external process reliability
- **Raised:** 2026-08-22T19:31:25Z
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
- **State history:** Open (2026-08-22T19:31:25Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Several benchmark paths use blocking child waits without a deadline: realtime.rs:379-383, bdrate.rs:322-355, bdrate-svc.rs:580-592 and 1017-1023, and encode-headtohead.rs:825-831. A hung external encoder or decoder therefore hangs an automated benchmark indefinitely. Soak already has a bounded kill-and-reap helper, showing the expected local contract. Route external tools through one bounded helper that closes stdin, kills and reaps on timeout, and reports the command role and retained diagnostics. Add fake-child timeout regressions for each call shape. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
