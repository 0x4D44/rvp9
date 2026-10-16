# RVP9-BUG-KILN-00045 - WebM EBML walk has unbounded recursion depth

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T174104Z-p18869-n374722000-c1 branch=task/bug-RVP9-BUG-KILN-00045-run-fix-20260730T174104Z-p18869-n374722000-c1 code=4e0e1b3332545b86cbfe72787b4ca23975562784 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The WebM EBML parser recurses through nested containers without a depth cap.

`crates/rvp9-io/src/container.rs:392`-`400` starts the walk with no depth budget. `walk_ebml` has no depth parameter at lines `404`-`410`, then recursively enters EBML/Segment, Cluster, and BlockGroup payloads at lines `436`-`449`. A malformed file with deeply nested recognized containers can grow call depth with input structure and exhaust stack before returning a clean parse error.

## Fix

Open. Add a conservative recursion-depth budget, or replace the recursive walk with an explicit stack and reject excessive nesting as malformed EBML.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. Existing EBML payload-size checks handle overruns, but they do not bound structural depth.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

`walk_ebml` now takes a `depth` parameter, incremented at every recognized-container recursion
(EBML/Segment, Cluster, BlockGroup -- the three sites named in the observation) and bounded by
`MAX_EBML_NESTING_DEPTH = 32` (`crates/rvp9-io/src/container.rs:589`, enforced at `:1077-1081`),
mirrored in the streaming reader at `:696` and `:822`. The companion `find_vp9_video_track` added
alongside is iterative, so it adds no new stack growth.
`webm_excessive_container_depth_is_rejected` (64 nested Segments) observed passing; without the
fix the fixture recurses happily and returns `Ok`, so `expect_err` fails.
Over-strict-guard check: 32 is far above the 3-4 levels any real WebM uses, and the 12 official
`.webm` conformance vectors fetched for this verification all parse and decode.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
