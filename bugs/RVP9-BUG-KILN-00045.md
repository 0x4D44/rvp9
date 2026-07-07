# RVP9-BUG-KILN-00045 - WebM EBML walk has unbounded recursion depth

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** io/webm packet reader
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`)

## Observation

The WebM EBML parser recurses through nested containers without a depth cap.

`crates/rvp9-io/src/container.rs:392`-`400` starts the walk with no depth budget. `walk_ebml` has no depth parameter at lines `404`-`410`, then recursively enters EBML/Segment, Cluster, and BlockGroup payloads at lines `436`-`449`. A malformed file with deeply nested recognized containers can grow call depth with input structure and exhaust stack before returning a clean parse error.

## Fix

Open. Add a conservative recursion-depth budget, or replace the recursive walk with an explicit stack and reject excessive nesting as malformed EBML.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. Existing EBML payload-size checks handle overruns, but they do not bound structural depth.
