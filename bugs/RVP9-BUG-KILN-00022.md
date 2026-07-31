# RVP9-BUG-KILN-00022 - Custom YV12 borders can overflow frame allocation geometry

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** core/frame allocation
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T155651Z-p22403-n646910000-c1 branch=task/bug-RVP9-BUG-KILN-00022-run-fix-20260730T155651Z-p22403-n646910000-c1 code=139c18b23ae3a29510092eee4049cded1c1c6b5e gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`BorderUsage::Custom(u32)` accepts arbitrary caller-supplied borders as long as they are multiples of 32 at `crates/rvp9-core/src/frame.rs:40`, and `Yv12Buffer::alloc` validates only visible frame dimensions and the multiple-of-32 rule at `crates/rvp9-core/src/frame.rs:206`.

The allocator then performs border arithmetic in `u32` before widening for allocation sizing: `width + 7` at `crates/rvp9-core/src/frame.rs:219`, `y_width + 2 * border_px` at `crates/rvp9-core/src/frame.rs:242`, and `y_height + 2 * border_px` at `crates/rvp9-core/src/frame.rs:244`. Huge aligned custom borders can pass validation and overflow or wrap before the later `usize::try_from(total)` check at `crates/rvp9-core/src/frame.rs:260`.

## Fix

Open. Validate custom border sizes before arithmetic and move allocation geometry math to checked `u64` or `usize` operations. Reject any configuration whose padded dimensions, strides, offsets, or total allocation cannot be represented safely. Add regressions for very large aligned custom borders.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Every arithmetic step named in the observation (width+7, y_width+2*border, y_height+2*border,
total) is now checked u64 arithmetic funnelling into one typed `AllocationTooLarge`
(`crates/rvp9-core/src/frame.rs:251-353`), and `vec![0u8; n]` was replaced with
`try_reserve_exact` so a huge-but-representable request errors instead of aborting on OOM.
`huge_aligned_custom_border_is_rejected`, `custom_border_with_overlarge_total_allocation_is_rejected`
and `ordinary_custom_border_still_allocates` all observed passing -- the third is the
over-strictness control, pinning that `Custom(64)` still allocates.
Over-strict-guard check (the RVP9-BUG-KILN-00020 failure mode): every in-tree
`BorderUsage::Custom` use was traced; only `frame.rs:785` and tests. The new error fires only
on genuine u32/u64 overflow.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
