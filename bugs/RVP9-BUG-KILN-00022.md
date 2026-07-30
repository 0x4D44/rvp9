# RVP9-BUG-KILN-00022 - Custom YV12 borders can overflow frame allocation geometry

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** core/frame allocation
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T115157Z-p49207-n263095000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00022-run-fix-20260730T115157Z-p49207-n263095000-c1
- **Owner base:** 45cce06e3031bc03c637f1f634aaee649f66daaf
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T11:51:57Z
- **Owner until:** 2026-07-30T13:51:57Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`)

## Observation

`BorderUsage::Custom(u32)` accepts arbitrary caller-supplied borders as long as they are multiples of 32 at `crates/rvp9-core/src/frame.rs:40`, and `Yv12Buffer::alloc` validates only visible frame dimensions and the multiple-of-32 rule at `crates/rvp9-core/src/frame.rs:206`.

The allocator then performs border arithmetic in `u32` before widening for allocation sizing: `width + 7` at `crates/rvp9-core/src/frame.rs:219`, `y_width + 2 * border_px` at `crates/rvp9-core/src/frame.rs:242`, and `y_height + 2 * border_px` at `crates/rvp9-core/src/frame.rs:244`. Huge aligned custom borders can pass validation and overflow or wrap before the later `usize::try_from(total)` check at `crates/rvp9-core/src/frame.rs:260`.

## Fix

Open. Validate custom border sizes before arithmetic and move allocation geometry math to checked `u64` or `usize` operations. Reject any configuration whose padded dimensions, strides, offsets, or total allocation cannot be represented safely. Add regressions for very large aligned custom borders.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review.
