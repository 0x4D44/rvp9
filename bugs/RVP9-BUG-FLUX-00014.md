# RVP9-BUG-FLUX-00014 — GOP byte-identity golden lags accepted production-parity stream

- **State:** Fixed
- **Priority:** Should
- **Severity:** Low
- **Area:** encoder tests
- **Raised:** 2026-08-09T22:47:42Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T064100Z-d33dd923
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00014-run-verify-20261019T064100Z-d33dd923
- **Owner base:** e63c45e329f20f6eac25774dd837a4e5fbc778ea
- **Owner fingerprint:** sha256:0bc48855526e8d62376bbc9f861ba42308b333fab213f57e5bd499bf68b91949
- **Owner since:** 2026-10-19T06:41:00Z
- **Owner until:** 2026-10-19T10:41:00Z
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
