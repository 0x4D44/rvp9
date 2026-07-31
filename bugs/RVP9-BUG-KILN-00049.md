# RVP9-BUG-KILN-00049 - rvp9enc --webm writes an IVF file

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T182426Z-p49376-n915512000-c1 branch=task/bug-RVP9-BUG-KILN-00049-run-fix-20260730T182426Z-p49376-n915512000-c1 code=3730bba2422b226ac8670be8016f628719b0b4e4 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`rvp9enc --webm` is presented as a supported WebM output flag, but the encode paths still always mux IVF.

The flag table stores `opts.webm = true` at `crates/rvp9-cli/src/bin/rvp9enc.rs:740`-`746`. The one-pass path then builds `IvfHeader::vp9` and calls `write_ivf` at `crates/rvp9-cli/src/bin/rvp9enc.rs:1335`-`1338`. The two-pass path does the same at `crates/rvp9-cli/src/bin/rvp9enc.rs:1405`-`1409`.

Expected: `--webm` should either select a real WebM muxer or fail loudly as unsupported. Actual: it silently writes an IVF container.

## Fix

Reject `--webm` until a WebM muxer exists, or add a WebM writer and route the output path through it. Add a behavioral CLI test that verifies the output container magic when `--webm` is supplied.

## Notes

This is not a duplicate of `RVP9-BUG-KILN-00044`, which tracks WebM input demuxing.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

This record allowed either a real muxer or a loud failure; the loud-failure branch was taken.
`--webm` is now `apply: None` with help text "WebM output (not supported yet)"
(`crates/rvp9-cli/src/bin/rvp9enc.rs:495-500`), and the `webm` field and its setter are deleted.
`args::apply` turns `None` into "rvp9enc: --webm is recognised but not supported yet", and
`render_help` derives `[not yet]` from the same field, so the silent-IVF outcome is structurally
gone.
Binary path: the rejection happens in `args::parse` before `run()`, i.e. before any input read or
`File::create`.
`enc_webm_rejected_before_creating_ivf_output` observed passing; it spawns the real binary via
`env!("CARGO_BIN_EXE_rvp9enc")` and asserts the `.webm` output file was never created.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
