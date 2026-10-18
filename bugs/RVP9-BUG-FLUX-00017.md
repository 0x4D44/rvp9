# RVP9-BUG-FLUX-00017 — build-libvpx reports failure after successful macOS build

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** scripts/libvpx build
- **Raised:** 2026-08-10T21:31:54Z
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
- **State history:** Open (2026-08-10T21:31:54Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T06:29:17Z, deltic:auto role=fix run=fix-20261016T061723Z-26c153c2 branch=task/bug-RVP9-BUG-FLUX-00017-run-fix-20261016T061723Z-26c153c2 code=eac3c6466881ef9a5f61903a2a6244cd8f55c6c4 gate=manual) -> Closed (2026-10-18T15:37:24Z, deltic:manual role=verify run=verify-20261018T153549Z-dc27b7ce verified_fix_run=fix-20261016T061723Z-26c153c2 verdict=close)

## Observation

On macOS arm64, bash scripts/build-libvpx.sh successfully builds build/libvpx/vpxenc and build/libvpx/vpxdec, prints both artifacts, then exits with status 1. Reproduce from an initialized pinned submodule with bash scripts/build-libvpx.sh. Expected: status 0 after both required tools exist. Actual: the final artifact loop leaves the script status equal to the false test for the absent Windows vpxdec.exe filename.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T15:37:24Z, deltic:auto run=verify-20261018T153549Z-dc27b7ce verified_fix_run=fix-20261016T061723Z-26c153c2 verdict=close)

Verifier note: Manual verify: Independent verification by Codex session 01a14fa5-deaa-7262-8da3-c14e43807e55 (did not author the fix), 2026-10-18, task base 09697b55 containing eac3c646. On native macOS arm64 initialized pinned v1.14.1 and ran deltic timeout 600 bash scripts/build-libvpx.sh with stdin closed: exit 0, vpxenc/vpxdec and spatial SVC artifacts built and listed. python3 scripts/test_build_libvpx.py -v selected 3 tests and passed, including configure/make failure propagation. Reversed only final artifact-loop if guard to original [ -f ... ] && ls: selected test_successful_build_reports_unix_artifa...

## Notes
