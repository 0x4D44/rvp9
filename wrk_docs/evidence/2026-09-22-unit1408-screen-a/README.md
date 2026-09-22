# Unit 1408 screen A — 16x16 variance inlining rejected

Candidate commit `59767dda` added legal plain `#[inline]` to
`var_sum_sse_16x16_inner`. Control was integrated trunk `9e882a46`. Both
Crucible x86-64 binaries produced identical IVF bytes for every sample.

The screen used the pinned 1920x1080@30/1, 60-frame Ducks, In To Tree, and Old
Town clips. It used CPU-used 6, CBR 1260 kb/s, one thread, tile-columns 0,
AQ0, noise sensitivity 0, one warm-up, five paired samples, and two reversed
clip-order rotations. The relaxed Crucible policy omitted `--max-load1`; the
recorded load samples are in `summary.txt` below.

| clip | paired candidate/control |
| --- | ---: |
| ducks_take_off | 0.998445x |
| in_to_tree | 0.989857x |
| old_town_cross | 1.018742x |
| corpus | **1.002275x** |

The corpus result is 0.23% slower, so this candidate is rejected and no source
change is being integrated. The exact harness manifest and per-sample CSV were
captured on Crucible; the compact result and hashes are in `summary.txt`.
