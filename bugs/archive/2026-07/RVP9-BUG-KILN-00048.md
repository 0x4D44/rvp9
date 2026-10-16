# RVP9-BUG-KILN-00048 - Container and raw CLI input limits still load whole files

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/container + cli limits
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T201410Z-p81249-n283615000-c1 branch=task/bug-RVP9-BUG-KILN-00048-run-fix-20260730T201410Z-p81249-n283615000-c1 code=767a650247d174461df59c010162e0c0afa30871 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The shipped decode and raw encode paths can load whole input files before extension rejection or `--limit` can bound the work.

`crates/rvp9-io/src/container.rs:129`-`140` reads the whole file with `std::fs::read` before dispatching on `.ivf` or `.webm`, so an unsupported large extension is fully resident-loaded before returning `UnknownExtension`. `rvp9dec` calls `read_packets` at `crates/rvp9-cli/src/bin/rvp9dec.rs:399` and applies `--limit` only after packet parsing and decode decisions at lines `451`-`460`, so a limited decode still pays the full container read and packet walk.

The raw encoder path has the same shape. `crates/rvp9-cli/src/bin/rvp9enc.rs:1481` reads the full raw file, then `chunks_exact(frame_size).take(max)` applies `--limit` at lines `1495`-`1496`.

The Y4M encoder path is not part of this bug: `rvp9enc` passes `--limit` into `read_cropped` at lines `1451` and `1460`, so it bounds the number of frames read even though the reader still has separate validation bugs.

## Fix

Open. Dispatch on extension before reading container bytes, add streaming or early-stop container readers for limited decode, and have raw input read only the frame bytes required by `--limit` when a limit is present.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. This is separate from harness-specific bounded-run bugs because it affects the shipped CLI and shared container reader.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Real streaming, not a cap-plus-warning -- all three limbs of the observation are addressed:
extension dispatch moved ahead of the file read (`crates/rvp9-io/src/container.rs:130-144`), a
new streaming `PacketReader` for IVF and WebM (`:152-316`), `rvp9dec` pulls packets on demand and
breaks the loop at the limit (`crates/rvp9-cli/src/bin/rvp9dec.rs:401`, `:432-437`, `:518`), and
the raw encoder computes a frame budget instead of reading the file whole
(`crates/rvp9-cli/src/bin/rvp9enc.rs:1077-1147`).
Binary path traced from `rvp9dec::run()` -- `read_packets` is replaced by `PacketReader::open`,
so the SHIPPED binary streams.
`dec_limit_does_not_read_trailing_bad_ivf_record`,
`enc_raw_limit_does_not_read_trailing_partial_frame`,
`read_packets_rejects_unknown_extension_before_filesystem_read`,
`packet_reader_ivf_stops_before_unread_malformed_tail` and
`packet_reader_webm_yields_video_blocks_incrementally` all observed passing. The two CLI tests
are strong discriminators: pre-fix the trailing garbage was read and rejected.
Residual noted, not blocking: `read_packets` still does a whole-file read for valid
`.ivf`/`.webm`, but its ~20 remaining callers are all bench/harness/test-only -- no shipped-CLI
residue.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
