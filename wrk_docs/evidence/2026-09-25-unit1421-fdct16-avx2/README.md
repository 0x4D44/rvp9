# Unit 1421 — AVX2 FDCT16 first pass

The exact Unit 1416 Crucible x86 profile attributes 395 of 40,989 direct
sample rows (0.964%) to FDCT16's first pass, plus 95 setup rows. The existing
first pass uses two SSE2 eight-lane batches. This candidate processes all
16 columns in one AVX2 batch and leaves the scalar second pass unchanged.

The x86 release differential compared AVX2 with SSE2 first-pass intermediates
and scalar final coefficients over random valid residuals, varied strides,
constant extremes, and alternating axes. The new test failed when an AVX2
store index was deliberately swapped, then passed when restored. The full x86
forward-transform group passed 41/41; AArch64 passed 46/46. Both encode
screens emitted one stable, byte-identical IVF hash per clip and arm.

On the pinned Crucible FDCT16 kernel ruler, alternating baseline/candidate
runs measured 238.15/221.01 ns and 241.84/219.01 ns per call. The output
hash was identical (`ed878a7063a67e69`). This is a repeatable 7–9% local
kernel gain, not a 7–9% encoder gain.

The initial Ducks-only 30-pair screen was a tie: 1.000812x paired
candidate/control (0.981326x separate medians), with severe scheduling
outliers. A fresh three-clip 30-pair-per-clip screen gave 0.983983x paired
corpus (0.992087x separate medians): Ducks 0.999812x, In-to-tree 0.967325x,
Old-town-cross 0.985081x paired. Both screens used CPU6, CBR 1260 kb/s,
one thread, 60 frames per clip, one warmup, and two rotations. The modest
whole-encode gain is noisy; the repeatable kernel gain and no observed
encode regression justify retaining it as a small step, not claiming parity.

`screen-a/` and `screen-b/` hold the exact harness manifests, raw paired rows,
summary ratios, encoder hashes, and clip hashes. The control executable was
the retained integrated `5b6822c0` encoder, SHA-256
`8d175c1bffba875f8381461a7be447335ad8ec77847ef83bb07b965474b29f6e`;
the candidate executable was SHA-256
`74cf144fa108cc8fd747fabf334250cb379506065898b1f610a4ef77aec3ed1a`.

The last direct pinned-libvpx comparison remains Unit 1415 at 1.230937x
RVP9/libvpx paired corpus. It predates this change; the expected impact is
well below the noise of that whole-encode comparison.
