# Unit 1413 — exact-zero inter TX selector screen

The candidate changes only the inter-luma TX selector on exact-zero input.
Its source patch SHA-256 is
`5bbe7e43b9c206ad1ae364866587378ba1d3d92908c896ba494e78a28efa355b`.
The candidate x86 release encoder SHA-256 is
`a6db771f310887a120fcca472c8efc8e191ad256f9bcd769531845d61464d610`;
the integrated-code control is
`adf2da0757c6d9a9f1f37f028159f476ddd5c3055629eb254b2490634935d677`.
The control was built at `cdbf7a6a`; shipped code did not change through
`2b1ae11f`, the candidate base.

The first complete paired screen (`unit1413-screen-b-*`) used three pinned
1080p, 60-frame clips, CPU6 CBR 1260 kbps, one thread, one warmup, three
pairs, and two rotations. It imposed no host-load ceiling. Every candidate
IVF hash matched its control and was stable. Paired median test/control
ratios were 1.776819 (Ducks), 1.079780 (Tree), and 1.027894 (Town), for
an equal-clip geometric mean of 1.254033. The raw rows show multi-fold
stalls on both arms, so this screen alone cannot resolve a small change.

The original five-pair/four-rotation attempt was stopped before it wrote
timing rows because its reused output filenames were misread as progress.
Its incomplete output files are not performance evidence.

The completed full screen (`unit1413-screen-c-*`) used five pairs and four
rotations, again with no host-load ceiling. All IVF hashes matched and were
stable. Paired median ratios were 1.046531 (Ducks), 0.990039 (Tree), and
1.026938 (Town), giving a 1.020899 corpus ratio: the candidate was about
2.1% slower. It won only 8/20, 11/20, and 9/20 pairs on those clips.
Host load ranged from 9.45 to 26.19 busy logical cores out of 32, and the
raw rows retain multi-fold stalls on both arms.

The separate-median corpus ratio was 0.977761, but those independent arm
medians do not preserve matched pairs under changing load. The paired
result is the decision measure. Neither complete screen shows a repeatable
gain, so the candidate was rejected. The exact source diff is preserved in
`unit1413-selector-candidate.patch.gz` (compressed SHA-256
`29f64492856727fbcaad21a6802ad10b914627ffed06fe5b2d0d59cb62580dff`).
No candidate code or new RVP9/libvpx timing is retained; the last direct
integrated ratio remains Unit 1410's 1.227734x.
