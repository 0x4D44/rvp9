# rvp9-core

Shared VP9 codec primitives for the [RVP9](https://github.com/0x4D44/rvp9)
encoder and decoder. This crate contains the bitstream and entropy machinery,
transforms, prediction, loop filtering, frame storage, motion search, and
scalar/SIMD kernels used by the higher-level crates.

Most applications should depend on
[`rvp9-encoder`](https://crates.io/crates/rvp9-encoder),
[`rvp9-decoder`](https://crates.io/crates/rvp9-decoder), or
[`rvp9-cli`](https://crates.io/crates/rvp9-cli) instead. The public API is
documented on [docs.rs](https://docs.rs/rvp9-core).

## Status

`0.2.0-alpha.3` is an 8-bit VP9 preview. The project currently qualifies
one-pass real-time encoding at CPU-used 6–8 and validates decoding against the
official VP9 conformance vectors. High-bit-depth support is only a scaffold.

RVP9 is BSD-3-Clause licensed. See the repository's
[licensing and patent notices](https://github.com/0x4D44/rvp9#licence)
before redistribution.
