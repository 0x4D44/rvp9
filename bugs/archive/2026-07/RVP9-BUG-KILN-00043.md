# RVP9-BUG-KILN-00043 - Y4M writer labels Yuv440 output as C420jpeg

- **State:** Closed
- **Priority:** Could
- **Severity:** Medium
- **Area:** io/y4m writer
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T181630Z-p36913-n108202000-c1 branch=task/bug-RVP9-BUG-KILN-00043-run-fix-20260730T181630Z-p36913-n108202000-c1 code=c61248417f9a696342e23430e53214f4a87138ae gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The public Y4M writer emits incorrect metadata for `ChromaFormat::Yuv440` in release builds.

`crates/rvp9-io/src/y4m.rs:245`-`249` documents that 4:4:0 has no standard Y4M tag and must be rejected before reaching `write_header`. The match arm at lines `274`-`280` only `debug_assert!`s that invariant, then falls back to `"C420jpeg"` in release. Direct public API callers can therefore write a stream tagged as 4:2:0 while the decoded planes are 4:4:0.

The CLI currently guards this at `crates/rvp9-cli/src/bin/rvp9dec.rs:543`, so the shipped `rvp9dec --y4m` path is protected. The shared API remains corrupting for any other caller.

## Fix

Open. Return `io::ErrorKind::InvalidInput` from `write_header` for `ChromaFormat::Yuv440`, and update the docs to make the rejection part of the function contract.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. Priority is `Could` because the current CLI has a caller-side guard, but the public API behavior is still wrong.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The `debug_assert!(false)` plus `"C420jpeg"` release fallback is gone, replaced by an
unconditional `Err(InvalidInput)` returned before any byte is written, and the doc comment now
states the rejection as contract (`crates/rvp9-io/src/y4m.rs:467-472`). Explicitly checked that
no `debug_assert` remains in this path -- the rejection fires in RELEASE, which was the whole
point of the report.
`y4m::tests::write_header_rejects_unrepresentable_yuv440_without_output` observed passing; it
asserts both the error kind and that the output buffer stayed empty.
Over-strict-guard check: every `Yuv440` reference across rvp9-cli / rvp9-io / rvp9-harness /
rvp9-bench was grepped; the only writer-side caller is `rvp9dec.rs:482`, which already rejects
4:4:0 for `--y4m`, so no in-tree caller expected success.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
