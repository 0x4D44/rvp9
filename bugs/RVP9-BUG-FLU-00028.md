# RVP9-BUG-FLU-00028 — Worker scope signals completion before releasing erased borrows

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** core/tile threading
- **Raised:** 2026-08-20T21:14:56Z
- **Discovery source:** Automation
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T070449Z-111f6d0f
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00028-run-verify-20261019T070449Z-111f6d0f
- **Owner base:** 77dfbdf4f597bf56ce9431c7469655e3717ff8ff
- **Owner fingerprint:** sha256:2684ad4f9dd6ca76c0e5c98cb10d13281e90fad8ff42f97d107371d07a481e7a
- **Owner since:** 2026-10-19T07:04:49Z
- **Owner until:** 2026-10-19T11:04:49Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-20T21:14:56Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-21T22:19:45Z, deltic:auto role=fix run=fix-20260821T221859Z-2d81c328 branch=task/bug-RVP9-BUG-FLU-00028-run-fix-20260821T221859Z-2d81c328 code=1c9d92d4 gate=manual) -> Fixed (2026-10-19T10:53:08Z, deltic:auto role=fix run=fix-20260821T221859Z-2d81c328 branch=task/bug-RVP9-BUG-FLU-00028-run-fix-20260821T221859Z-2d81c328 code=a3dcd00e0267330d79da47c17cbc394ed340a5cf gate=manual provenance-only=true reconciled-by=Codex)

## Observation

Miri deterministically reports undefined behavior in WorkerPool::scope: the completion message can wake the caller before the lifetime-erased worker wrapper is destroyed, so the scope deallocates its panic bookkeeping while a worker still holds a strongly protected shared reference. The 256-iteration repeated_scopes_release_internal_borrows_before_return regression reproduces the fault. Move completion signaling outside the erased closure and send only after that closure is consumed and dropped.

## Fix

Historical fix provenance is reconciled below. Independent behavioral verification remains pending.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `a3dcd00e0267330d79da47c17cbc394ed340a5cf` (fix(core): release worker borrows before completion).
The original reference `1c9d92d4` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `841bfc61e91ecd022c66361ff99e38de7216f2a5` contains the original reference and descends from this fixing commit.
WorkerJob changes from an erased closure alias to a struct with separate run and done_tx. execute consumes run under catch_unwind before sending completion; scope.rs removes send from the erased bookkeeping closure, and both queued/caller-participating jobs use execute. Adds named 256-iteration repeated_scopes_release_internal_borrows_before_return regression. Precisely matches the record observation.
Contemporary corroboration: wrk_journals/2026.08.20 - JRN - Stage 5 safety qualification.md:3 (Unit 1255 names bug28 and the exact ownership-order repair; added by this commit); wrk_docs/2026.08.22 - DAILY - KILN-Windows - Repo state report.md:90 (old reference and bug pairing).
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

## Notes
