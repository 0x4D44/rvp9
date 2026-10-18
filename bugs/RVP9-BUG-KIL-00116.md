# RVP9-BUG-KIL-00116 — BD-rate reports measured bitrates using the wrong FPS override

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** bench/bdrate reporting
- **Raised:** 2026-08-22T19:31:58Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T160158Z-3e26ceff
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00116-run-verify-20261018T160158Z-3e26ceff
- **Owner base:** bf20830b7553471f60fb27ca70900473661c1a4f
- **Owner fingerprint:** sha256:e4e4ae45cc565c42d31cb5e4c828766e9e42140fa3697eaf05134694f2f147b5
- **Owner since:** 2026-10-18T16:01:58Z
- **Owner until:** 2026-10-18T20:01:58Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:31:58Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T07:23:47Z, deltic:auto role=fix run=fix-20261016T070337Z-923fca11 branch=task/bug-RVP9-BUG-KIL-00116-run-fix-20261016T070337Z-923fca11 code=d7647377840b3de06c7b4734cf7963ba36d69296 gate=manual)

## Observation

The bdrate encoder command uses the requested fps value at crates/rvp9-bench/src/bin/bdrate.rs:273-307, but main derives fps_hz from the source Y4M header at lines 526-532 and passes that value to measured_bitrate_kbps at lines 576-580. When the fps option differs from the source header, printed and CSV realized bitrates use the wrong clip duration. Both curves receive the same multiplicative error, so the final BD-rate percentage is invariant; the defect is the reported RD-point bitrate. Measure duration from the effective encoder FPS or the written IVF header, and add an override-vs-source regression. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
