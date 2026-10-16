# RVP9-BUG-KIL-00089 — E1 reports success after an explicit libvpx leg fails

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/e1 block stats
- **Raised:** 2026-08-22T13:20:53Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T033134Z-e9981b6d
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00089-run-fix-20261016T033134Z-e9981b6d
- **Owner base:** f17a7114baa300efdf454c95dd4943696299a3b8
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T03:31:34Z
- **Owner until:** 2026-10-16T07:31:34Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T13:20:53Z, raised via `deltic bugs new`)

## Observation

When an explicitly supplied, existing `vpxenc` fails, times out, or encounters
setup/I/O errors, `run_libvpx` converts the failure to `None` at
`crates/rvp9-harness/src/bin/e1_block_stats.rs:1182-1245`. The caller at lines
2254-2274 passes that absence to the report renderer, which emits `n/a` data,
writes the report, and returns normally.

Omitting `--vpxenc` is an intentional rvp9-only mode. Supplying a tool creates
an expectation that the side-by-side leg ran. Expected: a failed requested leg
returns a non-zero error or an explicit failed result. Actual: a partial report
can be presented as a successful E.1 comparison.

## Fix

<unfixed — raised only>

## Notes

Represent optional absence separately from execution failure and propagate the
latter to `main`. Add a fake-child regression covering non-zero exit and timeout.
Confirmed statically; no child process was launched.
