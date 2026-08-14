# RVP9 v0.2.0-alpha.2

This second public evaluation build adds qualified two- and three-layer
spatial SVC and widens decoder conformance. It remains an alpha while the
complete cross-feature production-parity matrix is still open.

## Supported encoder surface

- One-pass real-time VP9 at CPU-used 6, 7, and 8.
- VBR and CBR rate control.
- Cyclic-refresh AQ mode 3 and content-aware denoising.
- Two- and three-layer temporal SVC.
- Two- and three-layer spatial SVC.
- 720p30, 1080p30, and 1080p60 targets.
- 8-bit profile-0 IVF output.

## Qualification evidence

- Reference implementation: libvpx v1.14.1 at revision
  `12f3a2ac603e8f10742105519e0cd03c3b8f71dd`.
- Decoder: all 4,816 displayed frames pass across all 307 pinned official
  vectors with native and forced-scalar dispatch at one and two threads.
- Spatial SVC: 48/48 curves and 120/120 layer gates pass the `+0.5%` BD-rate
  limit; the worst result is `+0.488431%`.
- Spatial SVC speed: all 18 corpus groups pass the `1.05x` relative gate, and
  all 48 latency rows pass their 720p30, 1080p30, and 1080p60 budgets.
- Every one of 1,008 exercised spatial full or extracted streams decodes under
  RVP9, pinned libvpx vpxdec, and ffmpeg. RVP9 and vpxdec output is
  byte-identical.

Reproduce the codec-specific gates with:

```bash
./scripts/qualify-conformance.sh
./scripts/qualify-alpha-encoder.sh
./scripts/qualify-reference-decoder.sh
```

## Known limits

- CPU-used 8 three-layer temporal-SVC relative speed still awaits final
  quiet-host confirmation.
- This release does not claim final `<=+0.5%` BD-rate or `<=1.05x` libvpx
  speed parity across the complete v1 matrix.
- The final combined CPU6--8 CBR/VBR, AQ, denoiser, temporal-SVC, and
  spatial-SVC acceptance matrix has not yet run from one commit.
- The supplied binary archive targets macOS on Apple Silicon. Other targets
  can build from source but are not release-qualified here.
- The macOS binary archive is unsigned and not notarized. macOS may require
  explicit user approval before running downloaded binaries.
- The release host did not have cargo-fuzz or the nightly Miri component.
  A direct 2.39-million-iteration parser panic smoke completed without a
  crash, but that is not sanitizer or Miri evidence.

The project is BSD-3-Clause licensed. Review `LICENSE`, `PATENTS`, and
`ATTRIBUTION.md` before redistribution.
