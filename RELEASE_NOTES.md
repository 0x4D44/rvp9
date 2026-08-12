# RVP9 v0.2.0-alpha.1

This is the first public evaluation build of RVP9. It provides an 8-bit
real-time VP9 encoder and decoder while production-parity work continues.

## Supported encoder surface

- One-pass real-time VP9 at CPU-used 6, 7, and 8.
- VBR and CBR rate control.
- Cyclic-refresh AQ mode 3 and content-aware denoising.
- Two- and three-layer temporal SVC.
- 720p30, 1080p30, and 1080p60 targets.
- 8-bit profile-0 IVF output.

## Qualification evidence

- Reference implementation: libvpx v1.14.1 at revision
  `12f3a2ac603e8f10742105519e0cd03c3b8f71dd`.
- Decoder: 4,796/4,796 displayed frames pass across the 306 alpha-supported
  official vector files, with native and forced-scalar dispatch. One-thread
  and two-thread decoded output is byte-identical.
- Encoder: 32/32 real-time configurations pass for 60 frames each. Every
  stream has the expected frame count and decodes byte-identically under
  RVP9, pinned libvpx vpxdec, and ffmpeg.
- The complete workspace test suite passes in debug and release modes.
  Formatting and clippy with warnings denied also pass.

Reproduce the codec-specific gates with:

```bash
./scripts/qualify-alpha-conformance.sh
./scripts/qualify-alpha-encoder.sh
```

## Known limits

- Spatial SVC is not supported. The alpha gate therefore excludes exactly
  `vp90-2-22-svc_1280x720_3.ivf`; the unfiltered official suite keeps this
  failure visible.
- CPU-used 8 three-layer temporal-SVC speed remains provisional.
- This release does not claim final `<=+0.5%` BD-rate or `<=1.05x` libvpx
  speed parity across the complete v1 matrix.
- The supplied binary archive targets macOS on Apple Silicon. Other targets
  can build from source but are not release-qualified here.
- The release host did not have cargo-fuzz or the nightly Miri component.
  A direct 2.39-million-iteration parser panic smoke completed without a
  crash, but that is not sanitizer or Miri evidence.

The project is BSD-3-Clause licensed. Review `LICENSE`, `PATENTS`, and
`ATTRIBUTION.md` before redistribution.
