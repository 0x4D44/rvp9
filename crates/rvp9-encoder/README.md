# rvp9-encoder

An 8-bit, one-pass real-time VP9 video encoder from the
[RVP9](https://github.com/0x4D44/rvp9) project. It targets WebRTC and
livestreaming workloads and follows libvpx's real-time encoder architecture.

The supported alpha surface covers CPU-used 6–8, CBR and VBR rate control,
cyclic-refresh AQ mode 3, content-aware denoising, and two- or three-layer
temporal and spatial SVC. The qualified targets are 720p30, 1080p30, and
1080p60. See the [project README](https://github.com/0x4D44/rvp9#alpha-scope)
for the production-quality and experimental quality boundaries.

```toml
[dependencies]
rvp9-encoder = "0.2.0-alpha.3"
```

The public API is documented on [docs.rs](https://docs.rs/rvp9-encoder).

## Status

`0.2.0-alpha.3` is ready for evaluation, not a final production-parity
release. It emits 8-bit profile-0 VP9; CPU-used 0–5, hardware encoding, and
high-bit-depth output are outside the current scope.

RVP9 is BSD-3-Clause licensed. See the repository's
[licensing and patent notices](https://github.com/0x4D44/rvp9#licence)
before redistribution.
