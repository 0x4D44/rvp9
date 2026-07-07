# RVP9-BUG-KILN-00049 - rvp9enc --webm writes an IVF file

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc container output
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`)

## Observation

`rvp9enc --webm` is presented as a supported WebM output flag, but the encode paths still always mux IVF.

The flag table stores `opts.webm = true` at `crates/rvp9-cli/src/bin/rvp9enc.rs:740`-`746`. The one-pass path then builds `IvfHeader::vp9` and calls `write_ivf` at `crates/rvp9-cli/src/bin/rvp9enc.rs:1335`-`1338`. The two-pass path does the same at `crates/rvp9-cli/src/bin/rvp9enc.rs:1405`-`1409`.

Expected: `--webm` should either select a real WebM muxer or fail loudly as unsupported. Actual: it silently writes an IVF container.

## Fix

Reject `--webm` until a WebM muxer exists, or add a WebM writer and route the output path through it. Add a behavioral CLI test that verifies the output container magic when `--webm` is supplied.

## Notes

This is not a duplicate of `RVP9-BUG-KILN-00044`, which tracks WebM input demuxing.
