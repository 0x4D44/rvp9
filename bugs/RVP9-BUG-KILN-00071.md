# RVP9-BUG-KILN-00071 — Legacy Encoder accepts dimensions that overflow derived geometry

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/synthetic API
- **Raised:** 2026-08-16T08:19:15Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261015T222840Z-dc59d6d8
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00071-run-fix-20261015T222840Z-dc59d6d8
- **Owner base:** 64062b59e6ea6322d62b7431249fe1a6d865ef88
- **Owner fingerprint:** -
- **Owner since:** 2026-10-15T22:28:40Z
- **Owner until:** 2026-10-16T02:28:40Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-16T08:19:15Z, raised via `deltic bugs new`)

## Observation

The public legacy Encoder validates only nonzero 64-aligned width and height at D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\lib.rs:339. Its keyframe path multiplies u32 mi_rows and mi_cols before converting to usize at lib.rs:729 and then allocates derived tile storage infallibly. A documented-valid configuration such as 524,288 by 524,288 therefore overflows in debug builds or wraps in release builds before a likely allocation panic. The modern StreamEncoder already applies a bounded luma envelope, so the two public entry points disagree.

## Fix

Apply the same maximum coded dimension and luma-pixel limits used by StreamEncoder. Compute every derived cell count and allocation size with checked arithmetic before encoding starts, returning InvalidConfig on failure. Add a regression that rejects a very large aligned configuration without allocating frame-sized storage.

## Notes
