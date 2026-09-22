# Unit 1410 — AVX2 EOB reduction

Candidate `2ebca99e` reduces the quantizer's inverse-scan EOB fallback in
16 signed `i16` lanes instead of widening 8 lanes to `i32`. It leaves the
quantization arithmetic unchanged. The control is integrated `f113f141`.

All screens used native Crucible release binaries, three pinned 1920×1080
60-frame clips, CPU-used 6, CBR 1260 kb/s, 30 fps, one thread, zero tile
columns, AQ0, noise sensitivity 0, one warmup, and two reversed clip orders.
The agreed relaxed Crucible policy omitted `--max-load1`. Every RVP9
candidate/control IVF hash matched, with one stable hash per clip and arm.

| Screen | Pairs per rotation | Paired candidate/control | Result |
| --- | ---: | ---: | --- |
| A | 5 | 1.000727× | effectively flat/slower |
| B | 3 | 0.996057× | faster on all three clips |
| C | 5 | 0.991624× | faster on all three clips |

Across the 26 paired samples per clip, the geometric mean of individual
candidate/control time ratios is 0.996041×, about a 0.40% gain. Screen A is
not hidden: its near-zero opposite sign limits confidence in the size of the
gain. Screens B and C independently favour the candidate on every clip. This
meets the agreed "every little helps" retention judgment without claiming a
large or precisely estimated speedup.

The manifests, raw samples, and summaries are in this directory and sibling
`2026-09-22-unit1410-screen-b/` and `-c/`. The x86 red-oracle record is
`red-oracle.txt`. The fresh pinned-libvpx comparison is in sibling
`2026-09-22-unit1410-libvpx/`.
