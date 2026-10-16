# RVP9-BUG-KIL-00109 — Soak accepts a zero-frame run as successful work

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/soak
- **Raised:** 2026-08-22T19:30:49Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T031436Z-34745546
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00109-run-fix-20261016T031436Z-34745546
- **Owner base:** e352388e97efd588d876749accb1da912d4ea583
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T03:14:36Z
- **Owner until:** 2026-10-16T07:14:36Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:30:49Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

crates/rvp9-bench/src/bin/soak.rs:146-164 accepts frames-per-iter zero. run_one_iteration at lines 375-459 then encodes and decodes no frames, and the equality check accepts decoded zero as the expected zero. With duration zero and iterations zero, the outer loop also has no terminating condition. Expected: each successful soak iteration exercises at least one frame and every configured run has a finite stop condition. Actual: empty work can report success or spin forever. Reject zero frames-per-iter and reject the duration-zero plus iterations-zero combination; require a positive decoded count. Add parser and empty-run regressions. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
