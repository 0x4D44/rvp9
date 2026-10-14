# rvp9-decoder

An 8-bit VP9 video decoder from the
[RVP9](https://github.com/0x4D44/rvp9) project. It implements the VP9
bitstream and decoding process and supports tile-threaded decoding.

All 4,816 displayed frames across the project's 307 pinned official VP9
conformance vectors pass under native and forced-scalar dispatch with one and
two decoder threads. The public API is documented on
[docs.rs](https://docs.rs/rvp9-decoder).

```toml
[dependencies]
rvp9-decoder = "0.2.0-alpha.3"
```

## Status

`0.2.0-alpha.3` is an evaluation release. It supports the project's qualified
8-bit profile-0 surface; high-bit-depth decoding is not available.

For IVF/WebM input and Y4M/raw output, install the companion
[`rvp9-cli`](https://crates.io/crates/rvp9-cli) package.

RVP9 is BSD-3-Clause licensed. See the repository's
[licensing and patent notices](https://github.com/0x4D44/rvp9#licence)
before redistribution.
