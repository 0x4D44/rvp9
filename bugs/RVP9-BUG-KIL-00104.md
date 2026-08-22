# RVP9-BUG-KIL-00104 — Picker motion test passes without checking its result

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/picker tests
- **Raised:** 2026-08-22T18:32:51Z
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
- **State history:** Open (2026-08-22T18:32:51Z, raised via `deltic bugs new`)

## Observation

The non-ignored test `picker_finds_motion_on_shifted_block` at `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\tests\diag_picker_direct.rs:22` builds a source shifted two rows and three columns, calls both picker entry points at `diag_picker_direct.rs:121` and `:134`, and prints every result. It performs no assertion. Its only oracle is a comment at `diag_picker_direct.rs:145` saying NEWMV should be close to `(16, 24)` in Q3 units.

Expected: the named test fails when the picker does not find the documented motion. Actual: any non-panicking result, including ZEROMV or an unrelated motion vector, reports green and gives the suite a false picker oracle.

## Fix

Assert the chosen mode and motion vector against a justified exact value or tolerance, and assert enough cost ordering to prove the fixture discriminates the expected winner. Deliberately perturb the shift or expected vector once to prove the regression turns red. If the file is intentionally diagnostic-only, mark it ignored and rename it so it does not claim a tested result. Estimated effort: small.

## Notes

Confirmed by the test-coverage lens and the lead's complete source read. No matching open bug was found. The test was not run during this static review.
