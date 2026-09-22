# Unit 1409 screen A — 16x16 SAD4D row striping rejected

The candidate at `7d66bf08` split the hot AVX2 16x16 four-reference SAD into
independent even and odd row accumulators. It left every other shape unchanged.
The control was integrated trunk `8a332069`; both binaries were built on
Crucible with Rust 1.95.0.

The screen used the pinned 1920x1080@30, 60-frame Ducks, In To Tree, and Old
Town clips at CPU-used 6, CBR 1260 kb/s, one thread, zero tile columns, AQ0,
noise sensitivity 0, one warmup, five paired samples, and two reversed clip
orders. The relaxed Crucible policy omitted `--max-load1`. Recorded load ranged
from 2.857871 to 9.984774 busy logical cores. All candidate and control IVF
hashes matched, with one stable hash per clip and arm.

| clip | paired candidate/control |
| --- | ---: |
| Ducks | 1.001625x |
| In To Tree | 0.990931x |
| Old Town | 1.015396x |
| corpus | **1.002601x** |

The candidate is 0.26% slower on the paired corpus. No second screen or
pinned-libvpx comparison was warranted. The source change and its temporary
test were removed; this evidence is the only retained result.

The harness manifest, raw samples, summary, and red-oracle record beside this
file give the exact inputs, binary hashes, commands, and outputs.
