# Ducks GOLDEN full-tree timing ceiling

The direct search-count gap is mostly GOLDEN, but the fractional refinement
itself is a small CPU target. Three exact Crucible runs measured **56.25–56.60
ms** inside all GOLDEN full-tree searches per 60-frame Ducks encode, against
**2,062–2,078 ms** of instrumented process CPU. Its gross timer share is
about **2.7%**. The timer includes two clock reads per search, possible
descheduling, and called variance work; this is attribution, **not** a
production speed measurement or a claim that those searches can be skipped.

| Pair | Counter off CPU (ms) | Counter on CPU (ms) | GOLDEN timer (ms) | All full-tree timer (ms) |
|---|---:|---:|---:|---:|
| 1 | 2,062.500 | 2,062.500 | 56.293 | 169.896 |
| 2 | 2,031.250 | 2,062.500 | 56.253 | 169.839 |
| 3 | 2,046.875 | 2,078.125 | 56.598 | 170.849 |

Each counter-on run counted the same **554,931** full-tree entries and
**190,447** GOLDEN entries. The first run's 16×16 GOLDEN bucket took 54.458
ms for 182,969 entries. Counter-on and counter-off outputs in all six runs
matched the canonical RVP9 Ducks IVF SHA-256
`8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`.
The complete process-CPU manifest and three counter-on bucket tables are
`runs.csv` and `ducks-{1,2,3}-on.csv`.

## Reproduction and limits

- Source: unintegrated diagnostic commits `37254c15` and `5e5c43a9`, applied
  as four hash-verified files to clean production source commit `48cab42d`.
  No diagnostic code is on production trunk.
- Crucible: AMD Ryzen 9 9950X3D, x86-64. Release `rvp9enc.exe` SHA-256:
  `16af332327d90eed7898edf33962ae99b4b9e10fb0439059fafcdcb090cbb3f0`.
- Pinned 60-frame Ducks Y4M SHA-256:
  `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`.
- Same canonical 60-frame realtime CBR command as the paired work census:
  `--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr
  --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0
  --lag-in-frames=0 --tune-content=default --aq-mode=0
  --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf`.
  Only `RVP9_SUBPEL_ORIGIN_CENSUS=0/1` changed within each pair.
- AArch64: 20 focused subpixel tests, CLI build, deliberate-red timing
  accumulator test, and five-frame exact on/off IVF comparison passed.
  x86-64: release build, focused counter/timer test, six exact IVF hashes,
  and count reconciliation passed.
- `.NET Process.TotalProcessorTime` reports process CPU in 15.625 ms
  increments here. `Instant` accumulates elapsed time around each full-tree
  entry and may include waiting as well as CPU. The counter-on process used
  0–31.25 ms more CPU than its adjacent counter-off run, so timing overhead
  is material relative to the 56 ms GOLDEN sum. Do not compare these process
  times to libvpx as a production speed ratio.

## Decision

Ducks would need about a **15.9%** RVP9 CPU reduction to reach 1.05× its
pinned libvpx process time. GOLDEN fractional refinement's gross ~2.7%
timer share is too small to close that gap alone, even under the unrealistic
assumption that all of it could be removed without changing output. A modest
kernel gain may still be worthwhile, but libvpx's GOLDEN admission policy is
not an exact-output optimization candidate. Shift the main search for parity
back to exact-IP whole-encoder hotspots; a separate libvpx gate census would
explain policy, not by itself make RVP9 faster while preserving output.
