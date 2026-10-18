# RVP9-BUG-FLUX-00017 — build-libvpx reports failure after successful macOS build

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** scripts/libvpx build
- **Raised:** 2026-08-10T21:31:54Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T153549Z-dc27b7ce
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00017-run-verify-20261018T153549Z-dc27b7ce
- **Owner base:** 9624a18d5b958dc0c075d6090c4ed76a30e107ce
- **Owner fingerprint:** sha256:9ee2b3701056a3f13a2d38a4f6bcdf3a5bd11d31c22c0074cbc622d514a77924
- **Owner since:** 2026-10-18T15:35:49Z
- **Owner until:** 2026-10-18T19:35:49Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-10T21:31:54Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T06:29:17Z, deltic:auto role=fix run=fix-20261016T061723Z-26c153c2 branch=task/bug-RVP9-BUG-FLUX-00017-run-fix-20261016T061723Z-26c153c2 code=eac3c6466881ef9a5f61903a2a6244cd8f55c6c4 gate=manual)

## Observation

On macOS arm64, bash scripts/build-libvpx.sh successfully builds build/libvpx/vpxenc and build/libvpx/vpxdec, prints both artifacts, then exits with status 1. Reproduce from an initialized pinned submodule with bash scripts/build-libvpx.sh. Expected: status 0 after both required tools exist. Actual: the final artifact loop leaves the script status equal to the false test for the absent Windows vpxdec.exe filename.

## Fix

<unfixed — raised only>

## Notes
