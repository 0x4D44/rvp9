# RVP9-BUG-KIL-00091 — check-lld-format can omit unreadable inputs and exit successfully

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** harness/check-lld-format
- **Raised:** 2026-08-22T13:20:53Z
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
- **State history:** Open (2026-08-22T13:20:53Z, raised via `deltic bugs new`)

## Observation

`check-lld-format` silently drops directory-entry errors via
`entries.flatten()` and treats a per-file `check_one` read failure as a warning
at `crates/rvp9-harness/src/bin/check_lld_format.rs:195-209`. The final failure
flag at lines 224-257 is computed only from successfully read reports. It can
therefore exit zero after omitting one or even every candidate LLD.

Expected: every discovered input is accounted for, and an unreadable input
makes the checker fail. Actual: the command can report that all checked files
are structurally sound while silently excluding files it could not inspect.

## Fix

<unfixed — raised only>

## Notes

Treat directory iteration and file-read errors as input failures and return a
non-zero exit. Add an injected/read-error regression at the directory-walk
boundary. Confirmed statically; no checker run was performed.
