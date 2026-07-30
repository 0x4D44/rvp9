# RVP9-BUG-KILN-00048 - Container and raw CLI input limits still load whole files

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/container + cli limits
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T201410Z-p81249-n283615000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00048-run-fix-20260730T201410Z-p81249-n283615000-c1
- **Owner base:** 4366d00a2757902d4c0f0a2282a1909d618e8f32
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T20:14:10Z
- **Owner until:** 2026-07-30T22:14:10Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`)

## Observation

The shipped decode and raw encode paths can load whole input files before extension rejection or `--limit` can bound the work.

`crates/rvp9-io/src/container.rs:129`-`140` reads the whole file with `std::fs::read` before dispatching on `.ivf` or `.webm`, so an unsupported large extension is fully resident-loaded before returning `UnknownExtension`. `rvp9dec` calls `read_packets` at `crates/rvp9-cli/src/bin/rvp9dec.rs:399` and applies `--limit` only after packet parsing and decode decisions at lines `451`-`460`, so a limited decode still pays the full container read and packet walk.

The raw encoder path has the same shape. `crates/rvp9-cli/src/bin/rvp9enc.rs:1481` reads the full raw file, then `chunks_exact(frame_size).take(max)` applies `--limit` at lines `1495`-`1496`.

The Y4M encoder path is not part of this bug: `rvp9enc` passes `--limit` into `read_cropped` at lines `1451` and `1460`, so it bounds the number of frames read even though the reader still has separate validation bugs.

## Fix

Open. Dispatch on extension before reading container bytes, add streaming or early-stop container readers for limited decode, and have raw input read only the frame bytes required by `--limit` when a limit is present.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. This is separate from harness-specific bounded-run bugs because it affects the shipped CLI and shared container reader.
