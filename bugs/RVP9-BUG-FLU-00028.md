# RVP9-BUG-FLU-00028 — Worker scope signals completion before releasing erased borrows

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** core/tile threading
- **Raised:** 2026-08-20T21:14:56Z
- **Discovery source:** Automation
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260820T211757Z-43f2433c
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00028-run-fix-20260820T211757Z-43f2433c
- **Owner base:** 5be5dba90f67d90c9ba5705236c3ab436bba0e26
- **Owner fingerprint:** -
- **Owner since:** 2026-08-20T21:17:57Z
- **Owner until:** 2026-08-21T00:17:51Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-20T21:14:56Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Miri deterministically reports undefined behavior in WorkerPool::scope: the completion message can wake the caller before the lifetime-erased worker wrapper is destroyed, so the scope deallocates its panic bookkeeping while a worker still holds a strongly protected shared reference. The 256-iteration repeated_scopes_release_internal_borrows_before_return regression reproduces the fault. Move completion signaling outside the erased closure and send only after that closure is consumed and dropped.

## Fix

<unfixed — raised only>

## Notes
