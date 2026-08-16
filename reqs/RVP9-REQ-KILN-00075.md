# RVP9-REQ-KILN-00075 — Reuse TPL mode-estimation scratch per analysis worker

- **State:** Draft
- **Priority:** Should
- **Area:** encoder/TPL
- **Raised:** 2026-08-16T08:19:48Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-16T08:19:48Z, raised via `deltic reqs new`)

## Statement

Move the predictor, source-difference, and coefficient buffers now allocated on every TPL block into reusable analysis scratch. D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\tpl\mode_est.rs:240 allocates all three vectors per call, and tpl\driver.rs:400 invokes that path for every block of every TPL window frame. A 1920 by 1088 frame therefore performs 97,920 heap allocations for these three buffers alone. Preserve mode decisions and add an allocation-count or capacity-reuse check.

## Notes
