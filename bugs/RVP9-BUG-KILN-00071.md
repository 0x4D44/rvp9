# RVP9-BUG-KILN-00071 — Legacy Encoder accepts dimensions that overflow derived geometry

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/synthetic API
- **Raised:** 2026-08-16T08:19:15Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T155721Z-0787c522
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00071-run-verify-20261018T155721Z-0787c522
- **Owner base:** 6197f93e8d40e1caa5ce2f20cd3473b3c2400423
- **Owner fingerprint:** sha256:21dcfda8246afe506dab715b91965f549214326b19a4dc9a89b88c0e7bc3bcee
- **Owner since:** 2026-10-18T15:57:21Z
- **Owner until:** 2026-10-18T19:57:21Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-16T08:19:15Z, raised via `deltic bugs new`) -> Fixed (2026-10-15T22:41:36Z, deltic:auto role=fix run=fix-20261015T222840Z-dc59d6d8 branch=task/bug-RVP9-BUG-KILN-00071-run-fix-20261015T222840Z-dc59d6d8 code=32a5c6147257e2358f107bd35c447594809ed2ee gate=manual)

## Observation

The public legacy Encoder validates only nonzero 64-aligned width and height at D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\lib.rs:339. Its keyframe path multiplies u32 mi_rows and mi_cols before converting to usize at lib.rs:729 and then allocates derived tile storage infallibly. A documented-valid configuration such as 524,288 by 524,288 therefore overflows in debug builds or wraps in release builds before a likely allocation panic. The modern StreamEncoder already applies a bounded luma envelope, so the two public entry points disagree.

## Fix

Apply the same maximum coded dimension and luma-pixel limits used by StreamEncoder. Compute every derived cell count and allocation size with checked arithmetic before encoding starts, returning InvalidConfig on failure. Add a regression that rejects a very large aligned configuration without allocating frame-sized storage.

## Notes
