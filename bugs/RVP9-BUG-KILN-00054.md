# RVP9-BUG-KILN-00054 - rvp9dec raw I420/YV12 output writes native non-420 chroma planes

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9dec raw output
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T183206Z-p60512-n528921000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00054-run-fix-20260730T183206Z-p60512-n528921000-c1
- **Owner base:** afadc576161d3ebfbb610c6babaf5896ce3640ab
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T18:32:06Z
- **Owner until:** 2026-07-30T20:32:06Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`)

## Observation

`rvp9dec --i420` and `--yv12` write the decoded buffer's native chroma planes, even for legal Profile-1 non-4:2:0 streams where the output is not I420 or YV12-sized.

The raw writers emit `y,u,v` or `y,v,u` directly at `crates/rvp9-cli/src/bin/rvp9dec.rs:556`-`566`. `visible_planes` derives chroma dimensions from the decoded buffer's native `chroma_width()` and `chroma_height()` at `crates/rvp9-cli/src/bin/rvp9dec.rs:573`-`590`. The decoder supports legal non-420 layouts at `crates/rvp9-decoder/src/lib.rs:1320`-`1331`, and `Yv12Buffer::chroma_width` documents that Yuv444/Yuv440 chroma width equals luma width at `crates/rvp9-core/src/frame.rs:362`-`370`.

Expected: `--i420`/`--yv12` should either reject non-420 frames or perform explicit chroma conversion/downsampling. Actual: it writes native 4:2:2, 4:4:0, or 4:4:4 plane sizes under I420/YV12 option names.

## Fix

Reject `--i420` and `--yv12` unless `buf.chroma() == ChromaFormat::Yuv420`, or add explicit chroma conversion. If native-planar output is desired, expose it under a separate name that does not claim I420/YV12 layout.

## Notes

This is distinct from `RVP9-BUG-KILN-00031`, which tracks a decoder Profile-1 AQ2 correctness gap, and from `RVP9-BUG-KILN-00043`, which tracks Y4M header labeling in the shared IO writer.
