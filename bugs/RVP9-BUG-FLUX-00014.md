# RVP9-BUG-FLUX-00014 — GOP byte-identity golden lags accepted production-parity stream

- **State:** Fixed
- **Priority:** Should
- **Severity:** Low
- **Area:** encoder tests
- **Raised:** 2026-08-09T22:47:42Z
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
- **State history:** Open (2026-08-09T22:47:42Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-09T22:54:15Z, deltic:auto role=fix run=fix-20260809T225328Z-p91450-n870271000-c1 branch=task/bug-RVP9-BUG-FLUX-00014-run-fix-20260809T225328Z-p91450-n870271000-c1 code=adf6c1b621170f6d9af9ecd8e7e6c52327f5e088 gate=manual)

## Observation

The w_gop_topology byte-identity test deterministically emits 23,767 bytes with FNV-1a 0xcb117524725cf5ff after the accepted production-parity stack, while its golden still expects the earlier 23,186-byte stream. The independent schedule-and-decode oracle passes all 40 frames, so the stale golden makes the workspace gate false-red rather than identifying a GOP topology failure. Refresh the byte golden with the observed repeatability and retained schedule oracle.

## Fix

<unfixed — raised only>

## Notes
