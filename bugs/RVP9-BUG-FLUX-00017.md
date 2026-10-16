# RVP9-BUG-FLUX-00017 — build-libvpx reports failure after successful macOS build

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** scripts/libvpx build
- **Raised:** 2026-08-10T21:31:54Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T061723Z-26c153c2
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00017-run-fix-20261016T061723Z-26c153c2
- **Owner base:** 03b247d2d1ecd60361ad9af8ae97d972b4faf6dd
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T06:17:23Z
- **Owner until:** 2026-10-16T10:17:23Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-10T21:31:54Z, raised via `deltic bugs new`)

## Observation

On macOS arm64, bash scripts/build-libvpx.sh successfully builds build/libvpx/vpxenc and build/libvpx/vpxdec, prints both artifacts, then exits with status 1. Reproduce from an initialized pinned submodule with bash scripts/build-libvpx.sh. Expected: status 0 after both required tools exist. Actual: the final artifact loop leaves the script status equal to the false test for the absent Windows vpxdec.exe filename.

## Fix

<unfixed — raised only>

## Notes
