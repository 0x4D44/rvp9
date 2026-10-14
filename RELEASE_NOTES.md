# RVP9 v0.2.0-alpha.4

This fourth public evaluation build is an x86-64 encoder performance release.
It lands the AVX2/SSE2 kernel and encoder hot-path work done since alpha.3,
plus decoder and tooling fixes. It is also the first release from the public
`github.com/0x4D44/rvp9` repository.

## Encoder performance (x86-64)

- AVX2 fast quantizer, with vectorised EOB scans and an all-zero bypass for
  TX8 and TX16 blocks.
- AVX2 SAD4D kernels specialised by block size.
- Vectorised 8x8 and 16x16 forward DCTs, picker Hadamard, SSE and quantizer
  metrics, integral projection, VBP min/max, frame high-frequency Laplacian,
  and inter residual subtraction.
- Shortcuts in the AVX2 sub-pixel variance passes for integer, identity and
  zero-vertical filters.
- An early proof of all-zero chroma coefficients that skips the inter
  transform-size selector, a zero-EOB transform-distortion path, and caches
  for A4 partition descent.
- A serial loop-filter walker that avoids copying padded rows.

Measured on one x86-64 Windows host with a profile-guided (PGO) build at
CPU-used 6, CBR, one thread, on 60-frame 1080p Ducks, Tree and Town: RVP9 used
1.03x the process CPU of pinned libvpx `vpxenc` (equal-clip geometric mean of
two 30-pair screens). Ducks remains 17% slower than libvpx; Tree and Town are
at or below parity. A plain `cargo build --release` without PGO is slower than
that figure.

The performance work does not change encoder output. alpha.4 and the published
alpha.3 produce byte-identical IVF files for Foreman (320x288) at CPU-used 6
CBR with AQ 3 and at CPU-used 8 VBR with the denoiser, and for a 720p
synthetic clip at CPU-used 7 CBR with two threads and two tile columns.

## Fixes

- Decoder: reject invalid tile-header geometry, and bound tile scratch memory
  by worker count instead of one full frame per tile.
- Mode decision: the 16x16 and 32x32 IDCTs now use SIMD dispatch.
- Tile-info writer rejects invalid geometry in release builds.
- Tests: vector-dependent tests no longer report green without fixtures, and
  the panic-safety inter-recovery test no longer accepts a decode failure.
- Bounds and narrowing checks on the rectangular SSE2 subtraction path.

## Qualification evidence

- Reference implementation: libvpx v1.14.1 at revision
  `12f3a2ac603e8f10742105519e0cd03c3b8f71dd`.
- Decoder: all 307 pinned official vectors pass with native and forced-scalar
  dispatch at one and two threads (`scripts/qualify-conformance.sh`).
- `cargo fmt --check`, `cargo clippy --workspace --all-targets -D warnings`,
  and `cargo test --workspace --release` pass on x86-64 Windows.
- Each public crate was packaged, built and tested from its tarball, and
  dry-run published before upload.

## Known limits

- Release qualification for this build covers x86-64 Windows. The AArch64
  paths are unchanged in kind but were not re-qualified for this release.
- The cross-decoder matrix against pinned `vpxdec` and the full encoder
  quality census were not rerun for this release; the last full runs are
  recorded for alpha.3. The byte-identical encodes above show the encoder
  output itself is unchanged on those configurations.
- Open low-severity SIMD findings (`RVP9-BUG-FLU-00032`, `-00034`, `-00035`)
  concern inputs the encoder does not generate: strides near `usize::MAX`
  and loop-filter `blimit` 255 (the encoder's maximum is 193).
- WebM and Y4M readers have open hardening bugs for crafted input
  (`bugs/`). Do not feed untrusted container files to `rvp9-io` yet.
- This release does not claim final `<=+0.5%` BD-rate parity for experimental
  VBR or temporal-SVC modes. Their explicit ceilings are unchanged.
- The encoder emits 8-bit profile 0. High-bit-depth kernels, CPU-used 0–5,
  hardware encoding, VP8/AV1, and two-pass production parity are outside the
  current scope.

The project is BSD-3-Clause licensed. Review `LICENSE`, `PATENTS`, and
`ATTRIBUTION.md` before redistribution.
