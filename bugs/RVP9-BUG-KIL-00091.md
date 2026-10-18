# RVP9-BUG-KIL-00091 — check-lld-format can omit unreadable inputs and exit successfully

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** harness/check-lld-format
- **Raised:** 2026-08-22T13:20:53Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T153844Z-879c82c8
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00091-run-verify-20261018T153844Z-879c82c8
- **Owner base:** 1bd933042a658d8d7280834526e990b08fef0c1a
- **Owner fingerprint:** sha256:0067c04cd8d5efb40bb03ce801eaf6ec4792430e4ab10846e0b7bc9415d9d543
- **Owner since:** 2026-10-18T15:38:44Z
- **Owner until:** 2026-10-18T19:38:44Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T13:20:53Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T06:16:50Z, deltic:auto role=fix run=fix-20261016T060707Z-0e7ce44a branch=task/bug-RVP9-BUG-KIL-00091-run-fix-20261016T060707Z-0e7ce44a code=225595e6c1c3dd3c190c23bdba4edeb9a7719c0d gate=manual)

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
