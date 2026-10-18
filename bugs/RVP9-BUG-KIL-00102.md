# RVP9-BUG-KIL-00102 — Lookahead peek panics on i32 minimum index

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** encoder/lookahead API
- **Raised:** 2026-08-22T18:32:49Z
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
- **State history:** Open (2026-08-22T18:32:49Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T07:32:36Z, deltic:auto role=fix run=fix-20261016T072410Z-43317dc5 branch=task/bug-RVP9-BUG-KIL-00102-run-fix-20261016T072410Z-43317dc5 code=66fce968fbb9419634e58123953c870f6413acbc gate=manual) -> Closed (2026-10-18T15:47:20Z, deltic:manual role=verify run=verify-20261018T154043Z-b834f5ae verified_fix_run=fix-20261016T072410Z-43317dc5 verdict=close)

## Observation

The public `LookaheadCtx::peek` contract says negative indices further back than `MAX_PRE_FRAMES` return `None` at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\lookahead.rs:293`. Its negative branch instead evaluates `-index` before the range check at `lookahead.rs:312`. Negating `i32::MIN` overflows: checked/debug builds panic, while ordinary wrapping release behavior can reach the intended `None` result after casting.

Expected: `peek(i32::MIN)` behaves like every other too-far-back index and returns `None` in every build profile. Actual: the safe public API has build-profile-dependent behavior and can panic before applying its documented bounds check.

## Fix

Compute the magnitude without signed negation, for example with `index.unsigned_abs()`, and reject magnitudes above `MAX_PRE_FRAMES` before slot arithmetic. Add an `i32::MIN` regression alongside the existing `peek(-2)` out-of-range case. Estimated effort: small.

### Verification summary (2026-10-18T15:47:20Z, deltic:auto run=verify-20261018T154043Z-b834f5ae verified_fix_run=fix-20261016T072410Z-43317dc5 verdict=close)

Verifier note: Manual verify: Independent session01a14fa5, not fixer; base09697b55 contains66fce968. Original i32::MIN peek observation: exact peek_out_of_range_indices_never_panic selects1, passes across empty/populated/popped states. Reverse unsigned_abs to (-index): selected1 fails exit101, negate overflow at lookahead.rs:313 and assertion peek(-2147483648) panicked with empty context. Exact bytes restored, git diff empty, same exact test passes. Checked unsigned magnitude rejects before slot arithmetic. Fmt/clippy pass. Workspace suite had transient F02 timeout-marker failure; exact retry and all30 F02 t...

## Notes

Confirmed independently by the security lens, the lead's source trace, and the devil's-advocate pass. No code or test was run during this static review.
