# rvp9-io

Container and raw-video I/O helpers for the
[RVP9](https://github.com/0x4D44/rvp9) toolchain. The crate provides IVF
reading and writing, WebM reading, Y4M 4:2:0 reading and writing, and the MD5
digest format used by libvpx's `vpxdec`.

The API is documented on [docs.rs](https://docs.rs/rvp9-io). Applications that
only need to encode or decode VP9 should normally use
[`rvp9-encoder`](https://crates.io/crates/rvp9-encoder) or
[`rvp9-decoder`](https://crates.io/crates/rvp9-decoder).

## Status

`0.2.0-alpha.3` is an evaluation release. The supported video pipeline is
8-bit VP9 profile 0 with 4:2:0 input and output.

RVP9 is BSD-3-Clause licensed. See the repository's
[licensing and patent notices](https://github.com/0x4D44/rvp9#licence)
before redistribution.
