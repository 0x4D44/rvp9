# Spatial VBR component-age correction — 2026-08-15

This focused curve validates the spatial VBR component-age correction on the
limiting 720p30 CPU8/S3 In To Tree slice. It is not speed evidence because the
host exceeded the declared `0.75` load bound.

- RVP9 source parent: `d073f286496611449fcfb923172cb12daa93ada1` plus the
  uncommitted component-age correction described below.
- Pinned libvpx: `12f3a2ac603e8f10742105519e0cd03c3b8f71dd`.
- Input SHA-256:
  `3053ddad95a66bb970685412d58ee807813319668a57cd5f8a6d00c07415a895`.
- Tested release encoder SHA-256:
  `c6a949d65d0f9a8a38a21582bf142ea0d917a57829c9307def16ac0869243ef9`.
- CSV SHA-256:
  `40cdb2c0c406dabe43ff264eb2e60b1392a71dbe78a1c9ee4b315e5828162f3e`.

## Result

| Layer | Retained baseline | Component-age fix | Gate |
| --- | ---: | ---: | --- |
| SL0 | `-6.9157%` | `-6.8233%` | pass |
| SL1 | `-5.0725%` | `-4.8364%` | pass |
| SL2 | `+0.8997%` | `-0.7297%` | pass |

Matched-bitrate anchor-minus-rvp9 PSNR means are `-0.2494`, `-0.1063`, and
`-0.0203 dB` for SL0/SL1/SL2. All 24 rvp9 full and extracted streams passed
rvp9dec, pinned vpxdec, and FFmpeg: 72 decoder runs with byte-identical
rvp9dec/vpxdec I420 output.

The matched CBR control remained exactly
`-4.935827/-0.480137/-0.391822%`; its diagnostic CSV SHA-256 was
`d1a074f2b00a05d3d3e2d1e6a1229a712975e6b5d4e3041c4c3824bf0fe436b7`.

## Change and reproduction

Rvp9 keeps one rate controller per spatial layer, so its local frame counter
was `1` on every layer's second component. Libvpx increments
`current_video_frame` once per coded spatial component. The old rvp9 schedule
therefore collapsed SL1 and SL2 frame 1 to `active_best == active_worst ==
103`; at 420 kbps those frames emitted 145,088 and 449,960 bits against 1,403
and 5,614-bit targets. Mapping the local counter to the global component index
moves libvpx's special frame-one branch to SL1 component 0 only. The corrected
SL1/SL2 frame-1 sizes are 2,336 and 5,536 bits.

Reproduce with the command shape in
`bench/results/spatial-vbr-cpu8-s3-in-tree-20260815.md`, substituting the
current `target/release/rvp9enc` and this CSV path. The exact grid remains
`420,630,840,1260,2100,3360` kbps with CPU8, S3, VBR, 60 frames, AQ0,
denoiser0, one thread, and inter-layer prediction mode 0.
