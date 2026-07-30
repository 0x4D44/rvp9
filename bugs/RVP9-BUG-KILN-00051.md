# RVP9-BUG-KILN-00051 - rvp9dec can truncate output before bitstream validation

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9dec output safety
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T182817Z-p54755-n888673000-c1 branch=task/bug-RVP9-BUG-KILN-00051-run-fix-20260730T182817Z-p54755-n888673000-c1 code=2e0a6ac6e122f4fab1a22a356b0b6742b4fc07e6 gate=manual)

## Observation

`rvp9dec` creates or truncates the output file after container packetization, but before VP9 packet parsing or frame decode succeeds.

`run()` reads container packet ranges at `crates/rvp9-cli/src/bin/rvp9dec.rs:399`-`400`, opens the output sink at `crates/rvp9-cli/src/bin/rvp9dec.rs:410`-`414`, and only then constructs the decoder at `crates/rvp9-cli/src/bin/rvp9dec.rs:418`-`421`. VP9 packet parsing and frame decode happen later at `crates/rvp9-cli/src/bin/rvp9dec.rs:432`-`446`. `open_output` uses `std::fs::File::create` at `crates/rvp9-cli/src/bin/rvp9dec.rs:612`-`616`, so an existing output path is truncated before a malformed VP9 payload can fail.

Expected: a decode that fails before producing output should not destroy an existing output file. Actual: a valid container with invalid VP9 data can clobber the destination before returning an error.

## Fix

Defer file creation until the first frame is ready to emit, or write to a temporary file and rename only on successful completion. Also construct any fallible decode resources before opening the output path.

## Notes

This is distinct from `RVP9-BUG-KILN-00048`, which tracks whole-file reads and `--limit` boundedness.
