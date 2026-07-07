# RVP9-BUG-KILN-00039 - Harness defaults can read corpus and tools outside the current worktree

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/path-resolution
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`)

## Observation

Harness helper defaults can silently use local data and tools outside the task worktree.

`crates/rvp9-harness/src/y4m.rs:29` probes `D:\language\rvp9-2\tests\corpus\testsrc_720p30.y4m`. Lines `48` and `50` probe main-clone and legacy Foreman paths. `crates/rvp9-harness/src/bin/e1_block_stats.rs:1058` defaults to `D:\language\rvp9-2\tests\corpus\foreman_cif.y4m`, and line `1061` defaults to `D:\language\libvpx-stock\vpxenc.exe`.

In a task worktree, those defaults can cross-contaminate validation by reading stale corpus files or tool binaries from a different checkout. That breaks the repo's worktree-first assumption that the current tree is the state being inspected.

## Fix

Open. Resolve default corpus paths from the current workspace or require explicit paths for out-of-tree data and tools. If a shared corpus/tool cache remains useful, make it an explicit environment variable or CLI argument and print the selected path and hash.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. This is related to `RVP9-BUG-KILN-00016`'s corpus drift concern, but it is a separate path-resolution defect in harness defaults.
