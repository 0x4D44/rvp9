# RVP9-BUG-KILN-00015 - High-bit-depth encoder configs are accepted but written as 8-bit streams

- **State:** Open
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`)

## Observation

`EncoderConfig::bit_depth` documents that non-`Bd8` settings cause the encoder to refuse the frame at `crates/rvp9-encoder/src/frame_loop.rs:233`. `StreamEncoder::new` validates dimensions, GOP, SVC, framerate, bitrate, and rate-control config around `crates/rvp9-encoder/src/frame_loop.rs:1275`, but it does not reject non-8-bit `bit_depth`.

The packet builders still hardcode `WireBitDepth::Eight` at `crates/rvp9-encoder/src/frame_loop.rs:4421` and `crates/rvp9-encoder/src/frame_loop.rs:4861`. In high-bit-depth feature builds, callers can request `Bd10` or `Bd12`, construct a stream encoder successfully, and receive an 8-bit stream.

## Fix

Open. Reject `config.bit_depth != BitDepth::Bd8` in `StreamEncoder::new` until high-bit-depth encoding is implemented, and add a feature-enabled regression for that rejection.

## Notes

This is feature-gated: default non-HBD builds cannot construct non-`Bd8` variants. The bug is the API contract mismatch when those variants are enabled.
