# RVP9-BUG-KIL-00099 — Position-relative I/O helpers seek to absolute offsets

- **State:** Fixed
- **Priority:** Could
- **Severity:** Medium
- **Area:** io/y4m and ivf public APIs
- **Raised:** 2026-08-22T14:34:26Z
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
- **State history:** Open (2026-08-22T14:34:26Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T06:06:19Z, deltic:auto role=fix run=fix-20261016T055916Z-e25b0ad6 branch=task/bug-RVP9-BUG-KIL-00099-run-fix-20261016T055916Z-e25b0ad6 code=e13faf8e88d165ba35419dbd81ee1c426f6b3c68 gate=manual)

## Observation

Two public helpers begin at the caller current cursor but later seek as if they began at byte zero. parse_header reads a Y4M header from the current File position at crates/rvp9-io/src/y4m.rs:188-215, then seeks to absolute bytes_consumed at :216 instead of start + bytes_consumed. IvfStreamWriter::new writes its header at the current Write+Seek cursor at crates/rvp9-io/src/container.rs:448-453, but finish patches absolute offset 24 at :490-494. A prefixed Y4M parse resumes at the wrong bytes; a prefixed IVF write corrupts earlier data and leaves its actual frame count zero. Neither public contract requires cursor zero. Fix by capturing each initial stream position, using checked relative offsets, and adding prefixed-file/prefixed-cursor regressions. Estimated effort: small. Confirmed by correctness and reliability reviewers, an independent security skeptic for IVF, and the lead; no tests or application were run.

## Fix

<unfixed — raised only>

## Notes
