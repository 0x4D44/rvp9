# rvp9-cli

Command-line VP9 encoder and decoder tools from the
[RVP9](https://github.com/0x4D44/rvp9) project:

- `rvp9enc` reads Y4M or raw YUV and writes 8-bit profile-0 VP9 in IVF.
- `rvp9dec` reads VP9 from IVF or WebM and writes Y4M or raw pixels.

Install both binaries from crates.io:

```console
cargo install rvp9-cli --version '=0.2.0-alpha.4' --locked
rvp9enc --help
rvp9dec --help
```

Basic forms are:

```console
rvp9enc [options] -o output.ivf input.y4m
rvp9dec [options] input.ivf
```

The encoder supports the project's one-pass real-time CPU-used 6–8 surface,
including CBR/VBR, cyclic-refresh AQ, denoising, and temporal/spatial SVC.
Some libvpx command-line options are shown for familiarity but fail clearly
when the corresponding behaviour is not implemented.

## Status

`0.2.0-alpha.4` is ready for evaluation, not a final production-parity
release. See the [project README](https://github.com/0x4D44/rvp9#alpha-scope)
for qualification evidence, experimental quality bounds, platform coverage,
and current limits.

RVP9 is BSD-3-Clause licensed. See the repository's
[licensing and patent notices](https://github.com/0x4D44/rvp9#licence)
before redistribution.
