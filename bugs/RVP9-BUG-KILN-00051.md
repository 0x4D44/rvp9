# RVP9-BUG-KILN-00051 - rvp9dec can truncate output before bitstream validation

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T182817Z-p54755-n888673000-c1 branch=task/bug-RVP9-BUG-KILN-00051-run-fix-20260730T182817Z-p54755-n888673000-c1 code=2e0a6ac6e122f4fab1a22a356b0b6742b4fc07e6 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`rvp9dec` creates or truncates the output file after container packetization, but before VP9 packet parsing or frame decode succeeds.

`run()` reads container packet ranges at `crates/rvp9-cli/src/bin/rvp9dec.rs:399`-`400`, opens the output sink at `crates/rvp9-cli/src/bin/rvp9dec.rs:410`-`414`, and only then constructs the decoder at `crates/rvp9-cli/src/bin/rvp9dec.rs:418`-`421`. VP9 packet parsing and frame decode happen later at `crates/rvp9-cli/src/bin/rvp9dec.rs:432`-`446`. `open_output` uses `std::fs::File::create` at `crates/rvp9-cli/src/bin/rvp9dec.rs:612`-`616`, so an existing output path is truncated before a malformed VP9 payload can fail.

Expected: a decode that fails before producing output should not destroy an existing output file. Actual: a valid container with invalid VP9 data can clobber the destination before returning an error.

## Fix

Defer file creation until the first frame is ready to emit, or write to a temporary file and rename only on successful completion. Also construct any fallible decode resources before opening the output path.

## Notes

This is distinct from `RVP9-BUG-KILN-00048`, which tracks whole-file reads and `--limit` boundedness.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

`open_output`/`File::create` no longer runs before the decode loop: the sink is initialised to
`None` (`crates/rvp9-cli/src/bin/rvp9dec.rs:412-415`) and opened only once a frame has survived
decode, skip and the chroma gate (`:493-494`). Container, packet-parse, decode and worker-spawn
failures therefore all leave the destination untouched. The observation's second ask -- construct
fallible decode resources first -- also holds: `PacketReader::open` at `:401` and
`Decoder::try_*` at `:417-421` both precede the sink.
`dec_invalid_vp9_preserves_existing_output_file` observed passing; it drives the real
`CARGO_BIN_EXE_rvp9dec` against a sentinel-filled output file and asserts byte-identical
survival.
Residual noted, not blocking: a failure AFTER the first frame is emitted still leaves a partial
file -- inherent to streaming output and outside this observation. See RVP9-BUG-KILN-00065 for
the inverse problem now present in `rvp9enc`.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
