# RVP9 v0.2.0-alpha.3

This third public evaluation build hardens the decoder and shared SIMD paths,
refreshes complete conformance and cross-decoder evidence, and documents the
quality boundary of every supported alpha mode. It also gives every public
crate a dedicated crates.io README, keywords, and categories.

## Supported encoder surface

- One-pass real-time VP9 at CPU-used 6, 7, and 8.
- VBR and CBR rate control.
- Cyclic-refresh AQ mode 3 and content-aware denoising.
- Two- and three-layer temporal SVC.
- Two- and three-layer spatial SVC.
- 720p30, 1080p30, and 1080p60 targets.
- 8-bit profile-0 IVF output.

The production-quality surface is single-stream CBR, denoiser CBR, and
spatial CBR. VBR and temporal SVC remain experimental previews with explicit
quality ceilings in `PLAN.md`.

## Qualification evidence

- Reference implementation: libvpx v1.14.1 at revision
  `12f3a2ac603e8f10742105519e0cd03c3b8f71dd`.
- Decoder: all 4,816 displayed frames pass across all 307 pinned official
  vectors with native and forced-scalar dispatch at one and two threads. The
  four modes perform 19,264 bit-exact frame checks.
- Cross-decoder: all 32 exercised RVP9 streams decode to identical pixels
  under RVP9, pinned libvpx `vpxdec`, and FFmpeg. The reverse 24-stream matrix
  requires RVP9 and `vpxdec` to agree byte-for-byte on pinned-`vpxenc` output.
- Quality: the complete fresh non-spatial and spatial censuses preserve the
  strict production-CBR gates. Experimental temporal CBR/VBR stay within
  `+5%`/`+35%`; experimental spatial VBR stays within `+25%` per important
  clip and layer.
- Safety: bounded AddressSanitizer parser and full-frame decoder fuzz
  campaigns and an encode/decode soak smoke completed without a crash or
  sanitizer report. Decoder reconstruction bounds, worker completion, plane
  provenance, and scalar-under-Miri dispatch received focused hardening.

Reproduce the codec-specific gates with:

```bash
./scripts/qualify-conformance.sh
./scripts/qualify-alpha-encoder.sh
./scripts/qualify-reference-decoder.sh
```

## Known limits

- The final quiet-host relative-speed and absolute-latency rerun is pending.
  Existing accepted timing evidence remains green, but this release does not
  claim the final complete-matrix `<=1.05x` result from one release commit.
- This release does not claim final `<=+0.5%` BD-rate parity for experimental
  VBR or temporal-SVC modes. Their explicit ceilings are unchanged.
- Long-duration billion-iteration fuzzing and the 72-hour soak are parked for
  the Intel fleet and are not claims of this release.
- Release qualification to date covers macOS on Apple Silicon. Other targets
  can build from source but are not release-qualified here.
- The encoder emits 8-bit profile 0. High-bit-depth kernels, CPU-used 0–5,
  hardware encoding, VP8/AV1, and two-pass production parity are outside the
  current scope.

The project is BSD-3-Clause licensed. Review `LICENSE`, `PATENTS`, and
`ATTRIBUTION.md` before redistribution.
