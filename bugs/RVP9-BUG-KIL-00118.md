# RVP9-BUG-KIL-00118 — Invalid-Unicode argv panics both CLI binaries on Unix

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/argument parsing
- **Raised:** 2026-08-22T20:44:01Z
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
- **State history:** Open (2026-08-22T20:44:01Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T03:51:26Z, deltic:auto role=fix run=fix-20261016T034349Z-5848b65e branch=task/bug-RVP9-BUG-KIL-00118-run-fix-20261016T034349Z-5848b65e code=2693687d89c9e9a1e9fb50946a3fda7e2f0cf934 gate=manual)

## Observation

Both binaries collect std::env::args into Vec<String>. On Unix, a valid OS argument or filename containing invalid UTF-8 makes the iterator panic before controlled parser error handling. Expected: preserve paths as OsString or return a normal diagnostic for invalid option text. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:881 and crates/rvp9-cli/src/bin/rvp9dec.rs:354; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
