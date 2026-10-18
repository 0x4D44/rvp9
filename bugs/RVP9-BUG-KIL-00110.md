# RVP9-BUG-KIL-00110 — Tile profiler exits successfully without valid profiling evidence

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** bench/decode-profile-tiles
- **Raised:** 2026-08-22T19:31:08Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T175426Z-7f0a0821
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00110-run-verify-20261018T175426Z-7f0a0821
- **Owner base:** 2b6141aabfec94f0fb0d104133348c487f8e447e
- **Owner fingerprint:** sha256:640ffc4e52f8678e7c73114d59a08797eb9544ac5f35204c80ed58ea90001b91
- **Owner since:** 2026-10-18T17:54:26Z
- **Owner until:** 2026-10-18T21:54:26Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:31:08Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T01:23:35Z, deltic:auto role=fix run=fix-20261016T011011Z-413aab2b branch=task/bug-RVP9-BUG-KIL-00110-run-fix-20261016T011011Z-413aab2b code=a74227e178e46016a2d5c628243b9dc183962664 gate=manual)

## Observation

crates/rvp9-bench/src/bin/decode-profile-tiles.rs:72-106 accepts zero runs or zero minimum timed frames and returns cpu MHz zero when the bench feature is off. main at lines 202-232 still exits successfully and says it wrote rows even when no run or no timed frame produced evidence. Cargo.toml:62-75 also documents enabling rvp9-decoder/profile-tiles directly, while the cpu_mhz cfg checks the bench crate feature, so the documented command can emit an invalid zero-MHz conversion note. Expected: success requires supported instrumentation and positive profile rows. Actual: empty or misconfigured runs claim success. Require positive arguments and timed rows, make unsupported instrumentation fatal, and align feature detection with the documented invocation. Add regressions for zero runs, zero samples, default features, and both supported feature spellings. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
