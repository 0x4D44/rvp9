# RVP9-BUG-KILN-00053 - rvp9enc one-pass mode buffers the full input and output

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T202432Z-p93542-n623704000-c1 branch=task/bug-RVP9-BUG-KILN-00053-run-fix-20260730T202432Z-p93542-n623704000-c1 code=ab172a3e648c74774106d18417d146249d3ee370 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5); residual split to RVP9-BUG-KILN-00065)

## Observation

The one-pass `rvp9enc` path keeps the whole input sequence and every encoded packet resident before writing the output file.

`run()` first receives all source frames as a `Vec` at `crates/rvp9-cli/src/bin/rvp9enc.rs:1248`-`1258`. The Y4M reader returns `Vec<YuvFrameBuf>` at `crates/rvp9-io/src/y4m.rs:139`-`145` and pushes every frame at `crates/rvp9-io/src/y4m.rs:174`-`225`. The raw path reads the entire input with `std::fs::read` at `crates/rvp9-cli/src/bin/rvp9enc.rs:1481` and pushes copied frame planes at `crates/rvp9-cli/src/bin/rvp9enc.rs:1494`-`1500`.

After that, one-pass encode allocates `packets: Vec<Vec<u8>>` at `crates/rvp9-cli/src/bin/rvp9enc.rs:1305`, pushes every packet at `crates/rvp9-cli/src/bin/rvp9enc.rs:1316`-`1331`, then builds a second vector of packet slices and writes IVF only at `crates/rvp9-cli/src/bin/rvp9enc.rs:1334`-`1338`.

Expected: one-pass encoding should stream frames and packet records where possible. Actual: large inputs can require memory proportional to full source frames plus full encoded output.

## Fix

Stream one-pass input frames into `StreamEncoder`, and add an IVF writer that writes frame records incrementally while patching the header count at the end. Keep whole-sequence residency only where the two-pass driver genuinely needs it.

## Notes

`RVP9-BUG-KILN-00048` covers raw whole-file reads and decode `--limit` behavior, and explicitly excludes the bounded Y4M path. This bug covers one-pass encode residency for Y4M frames and encoded packets.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Genuine streaming, not a cap. Both residency limbs the observation named are gone: no
`Vec<YuvFrameBuf>` of the sequence and no `Vec<Vec<u8>>` of packets in one-pass
(`crates/rvp9-cli/src/bin/rvp9enc.rs:841-950`, `:1051-1230`;
`crates/rvp9-io/src/container.rs:437-500`; `crates/rvp9-io/src/y4m.rs:138-390`). Two-pass keeps
whole-sequence residency exactly as this record permitted. Binary path traced from `run()` --
`open_input`, `IvfStreamWriter` over a `BufWriter<File>`, and a per-frame pull; the shipped path
changed, not just a helper.
`y4m::tests::incremental_reader_stops_before_unread_bad_frame_tag`,
`container::tests::stream_writer_patches_count_and_preserves_packets` and
`enc_passes_two_still_writes_complete_ivf` all observed passing.
Residual split to RVP9-BUG-KILN-00065: the one-pass output file is now created and
header-written BEFORE encoding starts, so a mid-encode failure leaves a partial IVF where
previously nothing was written -- the inverse of what RVP9-BUG-KILN-00051 fixed for `rvp9dec`.
Second residual noted, not blocking: no test pins the streaming property itself, so a revert to
full buffering would leave the CLI suite green.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
