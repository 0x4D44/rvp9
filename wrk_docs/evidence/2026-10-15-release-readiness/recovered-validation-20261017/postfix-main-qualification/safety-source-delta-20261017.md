# Exact-candidate unsafe-source delta screen

Candidate source is `fb5c0c917038db6c640b3dbc7c2b84b567233333`. This read-only
screen compares it with `ebca64b0d3c7359e96caf1e6250c921c18d30a3f`, the source
pin covered by the earlier bounded safety review.

`git diff --name-status ebca64b0d3c7359e96caf1e6250c921c18d30a3f fb5c0c917038db6c640b3dbc7c2b84b567233 -- crates scripts`
reported seven changed crate paths: six Rust files in the CLI and encoder,
plus the CLI README. The Rust changes are in `rvp9-cli/src/bin/rvp9enc.rs`,
`rvp9-cli/tests/cli_roundtrip.rs`, and encoder `denoiser.rs`, `frame_loop.rs`,
`lib.rs`, and `pre_analysis.rs`.

A zero-context diff-line search for additions or removals containing `unsafe`,
`SAFETY`, raw pointer types/conversions, `from_raw`, `transmute`, pointer
addition/offset, or `Send`/`Sync` impls found no matches. This is a lexical
screen for explicit unsafe seams, not a review of the changed algorithms or a
proof that safe code cannot panic or violate a contract at a downstream unsafe
call.

On exact source `fb5c0c91`, a broad `rg -n '\bunsafe\b'` search over the core,
IO, decoder and encoder Rust trees found 973 matching source lines across 98
files. Comments, declarations, tests and repeated references are included; the
number is not a reviewed-block count. It updates the older search-only count of
1,043 lines/100 files without changing the conclusion that the full unsafe
inventory remains incomplete.

The current-source Apple Silicon Miri tile-threading subset passed 23 tests,
with zero failures/ignores and 1,630 filtered. Dependent-library Miri remains
blocked by FLU-00037; native SIMD ASan remains blocked by FLU-00039, and this
screen does not cover parsers, full-frame decoding, native SIMD, fuzzing or
every unsafe block. No new memory-safety conclusion follows from the lexical
screen.
