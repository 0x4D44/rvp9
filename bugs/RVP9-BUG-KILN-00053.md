# RVP9-BUG-KILN-00053 - rvp9enc one-pass mode buffers the full input and output

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc boundedness
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

The one-pass `rvp9enc` path keeps the whole input sequence and every encoded packet resident before writing the output file.

`run()` first receives all source frames as a `Vec` at `crates/rvp9-cli/src/bin/rvp9enc.rs:1248`-`1258`. The Y4M reader returns `Vec<YuvFrameBuf>` at `crates/rvp9-io/src/y4m.rs:139`-`145` and pushes every frame at `crates/rvp9-io/src/y4m.rs:174`-`225`. The raw path reads the entire input with `std::fs::read` at `crates/rvp9-cli/src/bin/rvp9enc.rs:1481` and pushes copied frame planes at `crates/rvp9-cli/src/bin/rvp9enc.rs:1494`-`1500`.

After that, one-pass encode allocates `packets: Vec<Vec<u8>>` at `crates/rvp9-cli/src/bin/rvp9enc.rs:1305`, pushes every packet at `crates/rvp9-cli/src/bin/rvp9enc.rs:1316`-`1331`, then builds a second vector of packet slices and writes IVF only at `crates/rvp9-cli/src/bin/rvp9enc.rs:1334`-`1338`.

Expected: one-pass encoding should stream frames and packet records where possible. Actual: large inputs can require memory proportional to full source frames plus full encoded output.

## Fix

Stream one-pass input frames into `StreamEncoder`, and add an IVF writer that writes frame records incrementally while patching the header count at the end. Keep whole-sequence residency only where the two-pass driver genuinely needs it.

## Notes

`RVP9-BUG-KILN-00048` covers raw whole-file reads and decode `--limit` behavior, and explicitly excludes the bounded Y4M path. This bug covers one-pass encode residency for Y4M frames and encoded packets.
