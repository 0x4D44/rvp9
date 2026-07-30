# RVP9-BUG-KILN-00052 - rvp9dec --threads can panic after eager worker allocation

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** cli/rvp9dec threading
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T174453Z-p32667-n004904000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00052-run-fix-20260730T174453Z-p32667-n004904000-c1
- **Owner base:** 990c5c09d62b5f657780e02c662f0f41a75731c8
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T17:44:53Z
- **Owner until:** 2026-07-30T19:44:53Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`)

## Observation

`rvp9dec --threads N` accepts any `usize` that parses, then constructs a decoder that eagerly spawns `N` OS worker threads and panics if thread creation fails.

The CLI parses the user value at `crates/rvp9-cli/src/bin/rvp9dec.rs:226`-`238` and passes it into `Decoder::with_threads` at `crates/rvp9-cli/src/bin/rvp9dec.rs:418`-`420`. `Decoder::with_threads` constructs `WorkerPool::new(n)` at `crates/rvp9-decoder/src/lib.rs:744`-`750`. `WorkerPool::new` loops over `0..n` and calls `.expect("rvp9 worker thread spawn failed")` at `crates/rvp9-core/src/tile_threading/pool.rs:70`-`75`.

The signed Q.1.B HLD explicitly allowed an uncapped `--threads N` domain for libvpx parity at `wrk_docs/Track_Q/Q.1.B_threads_cli_HLD.md:17`-`20`, so the numeric domain alone is not the reported defect. The defect is that user input can drive eager resource allocation and an unrecoverable panic instead of a normal CLI error.

## Fix

Make worker-pool construction fallible and return a CLI error on spawn failure. Consider lazy spawning or a practical default cap as a separate design decision, but the minimal fix is to avoid panicking on user-provided `--threads`.

## Notes

`RVP9-BUG-KILN-00051` tracks the separate output-file truncation order. This bug covers the thread-construction failure mode.
