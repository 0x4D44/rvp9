# RVP9-BUG-KILN-00015 - High-bit-depth encoder configs are accepted but written as 8-bit streams

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/config validation
- **Raised:** 2026-07-05
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`) -> Fixed (2026-07-31, deltic:auto role=fix run=fix-20260730T231100Z-p73496-n304297000-c1 branch=task/bug-RVP9-BUG-KILN-00015-run-fix-20260730T231100Z-p73496-n304297000-c1 code=19d53ba2044adeeda32db78593cb2a3d402c73fc gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`EncoderConfig::bit_depth` documents that non-`Bd8` settings cause the encoder to refuse the frame at `crates/rvp9-encoder/src/frame_loop.rs:233`. `StreamEncoder::new` validates dimensions, GOP, SVC, framerate, bitrate, and rate-control config around `crates/rvp9-encoder/src/frame_loop.rs:1275`, but it does not reject non-8-bit `bit_depth`.

The packet builders still hardcode `WireBitDepth::Eight` at `crates/rvp9-encoder/src/frame_loop.rs:4421` and `crates/rvp9-encoder/src/frame_loop.rs:4861`. In high-bit-depth feature builds, callers can request `Bd10` or `Bd12`, construct a stream encoder successfully, and receive an 8-bit stream.

## Fix

Open. Reject `config.bit_depth != BitDepth::Bd8` in `StreamEncoder::new` until high-bit-depth encoding is implemented, and add a feature-enabled regression for that rejection.

## Notes

This is feature-gated: default non-HBD builds cannot construct non-`Bd8` variants. The bug is the API contract mismatch when those variants are enabled.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Rejection is unconditional and sited exactly where this record's Fix proposed
(`crates/rvp9-encoder/src/frame_loop.rs:1290`).
The regression test is behind a feature gate, so no default workspace run has ever executed it.
Ran it explicitly for this verification:
`cargo test -p rvp9-encoder --features highbitdepth --lib stream_encoder_rejects_unsupported_high_bit_depths`
-> `frame_loop::tests::stream_encoder_rejects_unsupported_high_bit_depths ... ok`, 1 passed.
Note for future passes: this test is ABSENT from a plain `cargo test --workspace` log; its
absence is not evidence of a missing fix, but it does mean the default gate never exercises it.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
