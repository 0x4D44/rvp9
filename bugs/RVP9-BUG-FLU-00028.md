# RVP9-BUG-FLU-00028 — Worker scope signals completion before releasing erased borrows

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** core/tile threading
- **Raised:** 2026-08-20T21:14:56Z
- **Discovery source:** Automation
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
- **State history:** Open (2026-08-20T21:14:56Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-21T22:19:45Z, deltic:auto role=fix run=fix-20260821T221859Z-2d81c328 branch=task/bug-RVP9-BUG-FLU-00028-run-fix-20260821T221859Z-2d81c328 code=1c9d92d4 gate=manual)

## Observation

Miri deterministically reports undefined behavior in WorkerPool::scope: the completion message can wake the caller before the lifetime-erased worker wrapper is destroyed, so the scope deallocates its panic bookkeeping while a worker still holds a strongly protected shared reference. The 256-iteration repeated_scopes_release_internal_borrows_before_return regression reproduces the fault. Move completion signaling outside the erased closure and send only after that closure is consumed and dropped.

## Fix

<unfixed — raised only>

## Notes
