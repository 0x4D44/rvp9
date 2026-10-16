# RVP9-BUG-KIL-00102 — Lookahead peek panics on i32 minimum index

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** encoder/lookahead API
- **Raised:** 2026-08-22T18:32:49Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T072410Z-43317dc5
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00102-run-fix-20261016T072410Z-43317dc5
- **Owner base:** a3954ad849af063b01c0c41400246e6d93c4bafb
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T07:24:10Z
- **Owner until:** 2026-10-16T11:24:10Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T18:32:49Z, raised via `deltic bugs new`)

## Observation

The public `LookaheadCtx::peek` contract says negative indices further back than `MAX_PRE_FRAMES` return `None` at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\lookahead.rs:293`. Its negative branch instead evaluates `-index` before the range check at `lookahead.rs:312`. Negating `i32::MIN` overflows: checked/debug builds panic, while ordinary wrapping release behavior can reach the intended `None` result after casting.

Expected: `peek(i32::MIN)` behaves like every other too-far-back index and returns `None` in every build profile. Actual: the safe public API has build-profile-dependent behavior and can panic before applying its documented bounds check.

## Fix

Compute the magnitude without signed negation, for example with `index.unsigned_abs()`, and reject magnitudes above `MAX_PRE_FRAMES` before slot arithmetic. Add an `i32::MIN` regression alongside the existing `peek(-2)` out-of-range case. Estimated effort: small.

## Notes

Confirmed independently by the security lens, the lead's source trace, and the devil's-advocate pass. No code or test was run during this static review.
