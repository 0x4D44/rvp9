# RVP9-BUG-FLUX-00014 — GOP byte-identity golden lags accepted production-parity stream

- **State:** Open
- **Priority:** Should
- **Severity:** Low
- **Area:** encoder tests
- **Raised:** 2026-08-09T22:47:42Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260809T225328Z-p91450-n870271000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00014-run-fix-20260809T225328Z-p91450-n870271000-c1
- **Owner base:** 3c88cfba9c9a84d97514cd7f399c56e5783f5bad
- **Owner fingerprint:** -
- **Owner since:** 2026-08-09T22:53:28Z
- **Owner until:** 2026-08-10T00:53:28Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-09T22:47:42Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

The w_gop_topology byte-identity test deterministically emits 23,767 bytes with FNV-1a 0xcb117524725cf5ff after the accepted production-parity stack, while its golden still expects the earlier 23,186-byte stream. The independent schedule-and-decode oracle passes all 40 frames, so the stale golden makes the workspace gate false-red rather than identifying a GOP topology failure. Refresh the byte golden with the observed repeatability and retained schedule oracle.

## Fix

<unfixed — raised only>

## Notes
