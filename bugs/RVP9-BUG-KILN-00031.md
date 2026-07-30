# RVP9-BUG-KILN-00031 - Profile-1 AQ2 non-420 segmentation decode remains mismatched and untracked

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** decoder/profile1 segmentation
- **Raised:** 2026-07-07
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T163444Z-p94241-n760521000-c1 branch=task/bug-RVP9-BUG-KILN-00031-run-fix-20260730T163444Z-p94241-n760521000-c1 code=4de7766b06b78cc8e7b7ddb70a9d60252ec0cdfd gate=manual)

## Observation

`crates/rvp9-harness/tests/profile1_decode.rs:337` documents a real decoder bug in the Profile-1 AQ2 path: cyclic-refresh segmentation activates segment-id-based quantization and the non-420 decode path produces incorrect output. The comments record 9/12 mismatches on `foreman_c444_aq2.ivf` and 8/12 mismatches on `testsrc_c444_720p_aq2.ivf`.

Both follow-up tests remain ignored at `crates/rvp9-harness/tests/profile1_decode.rs:349` and `crates/rvp9-harness/tests/profile1_decode.rs:360`, and they still assert `mismatched > 0` rather than gating the expected zero-mismatch behavior.

`RVP9-BUG-KILN-00007` closed the kfonly Profile-1 ignored-test issue, but its notes explicitly leave the AQ2 portion open/unverified. Because that bug is Closed, the remaining AQ2 decoder mismatch is not in the Open bug queue.

## Fix

Open. Root-cause and fix the non-420 AQ2 segmentation decode mismatch, then convert both AQ2 follow-ups to zero-mismatch gates and move the fixtures into the normal Profile-1 vector set. Keep the bug Open until the ignored tests are either active or replaced by an equivalent default oracle.

## Notes

Confirmed by two skeptic passes in the 2026-07-07 decoder review; the third skeptic agreed the mismatch is real but considered the scratchpad note enough. It is filed because the fix loop drains `bugs/`, not scratchpad entries or Closed bug notes.
