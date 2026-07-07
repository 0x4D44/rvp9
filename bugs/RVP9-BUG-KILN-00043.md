# RVP9-BUG-KILN-00043 - Y4M writer labels Yuv440 output as C420jpeg

- **State:** Open
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`)

## Observation

The public Y4M writer emits incorrect metadata for `ChromaFormat::Yuv440` in release builds.

`crates/rvp9-io/src/y4m.rs:245`-`249` documents that 4:4:0 has no standard Y4M tag and must be rejected before reaching `write_header`. The match arm at lines `274`-`280` only `debug_assert!`s that invariant, then falls back to `"C420jpeg"` in release. Direct public API callers can therefore write a stream tagged as 4:2:0 while the decoded planes are 4:4:0.

The CLI currently guards this at `crates/rvp9-cli/src/bin/rvp9dec.rs:543`, so the shipped `rvp9dec --y4m` path is protected. The shared API remains corrupting for any other caller.

## Fix

Open. Return `io::ErrorKind::InvalidInput` from `write_header` for `ChromaFormat::Yuv440`, and update the docs to make the rejection part of the function contract.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. Priority is `Could` because the current CLI has a caller-side guard, but the public API behavior is still wrong.
