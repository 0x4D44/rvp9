# RVP9-BUG-KILN-00069 — ALTREF panics on legal partial-height streams

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** encoder/altref geometry
- **Raised:** 2026-08-16T08:18:59Z
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
- **State history:** Open (2026-08-16T08:18:59Z, raised via `deltic bugs new`) -> Fixed (2026-10-15T22:28:21Z, deltic:auto role=fix run=fix-20261015T222031Z-c71ab272 branch=task/bug-RVP9-BUG-KILN-00069-run-fix-20261015T222031Z-c71ab272 code=4780fcc841d9bba4f467bfaf40dd82b5bcc0365f gate=manual)

## Observation

StreamEncoder accepts every positive even height, but the enabled ALTREF path calls vp9_temporal_filter, which asserts 32-row alignment. At D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\frame_loop.rs:2117, 720p and 1080p are valid; D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\temporal_filter.rs:1528 panics for both. Heights divisible by 32 but not 64 can instead reach the padded reference-pool assertion because the filtered reconstruction is allocated at visible height. The existing ALTREF tests use only 64- or 128-row frames.

## Fix

Handle visible edge macroblocks in the temporal filter, and allocate the ARF reconstruction with the same padded storage height as the reference pool. Add delayed-encode and flush regressions for 1280×720, 1920×1080, and a height divisible by 32 but not 64. The tests must reach ARF emission and decode the resulting superframe.

## Notes
