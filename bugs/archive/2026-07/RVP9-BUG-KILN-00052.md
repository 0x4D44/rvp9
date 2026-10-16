# RVP9-BUG-KILN-00052 - rvp9dec --threads can panic after eager worker allocation

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** cli/rvp9dec threading
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T174453Z-p32667-n004904000-c1 branch=task/bug-RVP9-BUG-KILN-00052-run-fix-20260730T174453Z-p32667-n004904000-c1 code=2a574cf62bb5502a9eba733338adcb5e93826704 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`rvp9dec --threads N` accepts any `usize` that parses, then constructs a decoder that eagerly spawns `N` OS worker threads and panics if thread creation fails.

The CLI parses the user value at `crates/rvp9-cli/src/bin/rvp9dec.rs:226`-`238` and passes it into `Decoder::with_threads` at `crates/rvp9-cli/src/bin/rvp9dec.rs:418`-`420`. `Decoder::with_threads` constructs `WorkerPool::new(n)` at `crates/rvp9-decoder/src/lib.rs:744`-`750`. `WorkerPool::new` loops over `0..n` and calls `.expect("rvp9 worker thread spawn failed")` at `crates/rvp9-core/src/tile_threading/pool.rs:70`-`75`.

The signed Q.1.B HLD explicitly allowed an uncapped `--threads N` domain for libvpx parity at `wrk_docs/Track_Q/Q.1.B_threads_cli_HLD.md:17`-`20`, so the numeric domain alone is not the reported defect. The defect is that user input can drive eager resource allocation and an unrecoverable panic instead of a normal CLI error.

## Fix

Make worker-pool construction fallible and return a CLI error on spawn failure. Consider lazy spawning or a practical default cap as a separate design decision, but the minimal fix is to avoid panicking on user-provided `--threads`.

## Notes

`RVP9-BUG-KILN-00051` tracks the separate output-file truncation order. This bug covers the thread-construction failure mode.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The exact panic site named in the observation (`.expect("rvp9 worker thread spawn failed")`) is
now confined to the infallible `new()` convenience wrapper, and the user-driven path no longer
reaches it: `Decoder::try_with_threads`/`try_new` with `map_err` to a CLI error
(`crates/rvp9-cli/src/bin/rvp9dec.rs:417-421`), plus fallible `try_new`, `try_reserve_exact` and
join-and-return cleanup of partially spawned workers
(`crates/rvp9-core/src/tile_threading/pool.rs:83-118`, `lf_pool.rs:126-165`).
`threads_unallocatable_count_returns_error_without_panicking` (CLI, allocation-failure limb),
`tile_threading::pool::construction_tests::worker_pool_spawn_failure_is_returned` and
`tile_threading::lf_pool::tests::spawn_failure_is_returned` (injected spawn failure -- the limb
the CLI test cannot reach) all observed passing. `--threads` is still uncapped, so `usize::MAX`
genuinely reaches `try_with_threads`; the test asserts stderr contains "creating decoder worker
threads" and does NOT contain "panicked". Pre-fix that input aborted the process.
Residual noted, not blocking: `WorkerPool::new`/`LfWorkerPool::new`/`Decoder::new` still panic by
design, and no shipped CLI path uses them for user input.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
