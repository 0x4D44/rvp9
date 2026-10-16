# RVP9-BUG-KILN-00039 - Harness defaults can read corpus and tools outside the current worktree

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T193433Z-p28955-n240937000-c1 branch=task/bug-RVP9-BUG-KILN-00039-run-fix-20260730T193433Z-p28955-n240937000-c1 code=c49fa959fa4ad4ca06e94ed0ba7989eaea5181e1 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

Harness helper defaults can silently use local data and tools outside the task worktree.

`crates/rvp9-harness/src/y4m.rs:29` probes `D:\language\rvp9-2\tests\corpus\testsrc_720p30.y4m`. Lines `48` and `50` probe main-clone and legacy Foreman paths. `crates/rvp9-harness/src/bin/e1_block_stats.rs:1058` defaults to `D:\language\rvp9-2\tests\corpus\foreman_cif.y4m`, and line `1061` defaults to `D:\language\libvpx-stock\vpxenc.exe`.

In a task worktree, those defaults can cross-contaminate validation by reading stale corpus files or tool binaries from a different checkout. That breaks the repo's worktree-first assumption that the current tree is the state being inspected.

## Fix

Open. Resolve default corpus paths from the current workspace or require explicit paths for out-of-tree data and tools. If a shared corpus/tool cache remains useful, make it an explicit environment variable or CLI argument and print the selected path and hash.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. This is related to `RVP9-BUG-KILN-00016`'s corpus drift concern, but it is a separate path-resolution defect in harness defaults.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Every path named in the observation is gone. Corpus probes now resolve solely from
`env!("CARGO_MANIFEST_DIR")/../../tests/corpus` (`crates/rvp9-harness/src/y4m.rs:10-26`, `:34-38`,
`:50-54`), and the `--vpxenc` default is removed, so with no path supplied the libvpx leg is
skipped rather than silently running an out-of-tree binary
(`crates/rvp9-harness/src/bin/e1_block_stats.rs:1050`, `:1177`, `:1178`). A grep for absolute
`D:\language` paths across all `*.rs` found only usage examples in rvp9-bench doc comments.
`y4m::tests::corpus_candidates_stay_inside_this_workspace` and
`tests::vpxenc_requires_an_explicit_path` both observed passing.
Independent corroboration from this verification: this worktree started with no `tests/corpus`
and no `build/libvpx`, and the harness correctly found nothing rather than reaching into the
main clone at `D:\language\rvp9-2`, which has both.
Residual noted, not blocking: `CARGO_MANIFEST_DIR` bakes a compile-time absolute path, so a
harness binary built in worktree A and run from clone B still reads A's corpus. That matches the
commit's stated intent but is not the same as "the cwd you are in".

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
