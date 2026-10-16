# RVP9-BUG-FLUX-00017 — build-libvpx reports failure after successful macOS build

- **State:** Fixed
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
- **State history:** Open (2026-08-10T21:31:54Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T06:29:17Z, deltic:auto role=fix run=fix-20261016T061723Z-26c153c2 branch=task/bug-RVP9-BUG-FLUX-00017-run-fix-20261016T061723Z-26c153c2 code=eac3c6466881ef9a5f61903a2a6244cd8f55c6c4 gate=manual)

## Observation

On macOS arm64, bash scripts/build-libvpx.sh successfully builds build/libvpx/vpxenc and build/libvpx/vpxdec, prints both artifacts, then exits with status 1. Reproduce from an initialized pinned submodule with bash scripts/build-libvpx.sh. Expected: status 0 after both required tools exist. Actual: the final artifact loop leaves the script status equal to the false test for the absent Windows vpxdec.exe filename.

## Fix

<unfixed — raised only>

## Notes
