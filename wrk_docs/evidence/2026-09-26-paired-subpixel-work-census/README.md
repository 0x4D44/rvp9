# Paired subpixel work census (Crucible, 2026-09-26)

This is a **work-count diagnostic**, not a performance result. It compares actual
fractional variance calls and full-tree search entries for three 60-frame,
1080p30 CPU6 CBR clips on x86-64 Crucible. The latest separate, uninstrumented
15-pair RVP9/pinned-libvpx process-CPU ratio remains **1.134647×** equal-clip
(Ducks 1.249014, Tree 1.128710, Town 1.036176); see
`../2026-09-26-avx2-all-zero-eob-fast-return/README.md`.

| Clip | RVP9 searches | libvpx searches | RVP9/libvpx | RVP9 fractional probes/search | libvpx fractional probes/search |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ducks | 554,931 | 370,507 | 1.4978× | 9.99999 | 10.00000 |
| Tree | 124,474 | 177,041 | 0.7031× | 9.99998 | 9.99995 |
| Town | 120,450 | 200,604 | 0.6004× | 10.00000 | 9.99999 |

On Ducks, 530,902 RVP9 searches are 16×16. The two encoders have nearly the
same probes per search, so the Ducks fractional-work excess is primarily *more
search entries*, not more probes inside one search. The reverse entry ratios on
Tree and Town rule out a global RVP9 subpixel-search-frequency problem. Search
counts do not prove that any search is redundant or that removing it preserves
output. Nor do counts price a probe in CPU time; shape mix, interpolation paths,
candidate gates and surrounding work differ.

The RVP9 diagnostic starts from the code-equivalent Crucible commit `48cab42d`
and applies the diffs archived as `rvp9-subpixel-census.patch.txt` plus
`rvp9-search-entry-census.patch.txt`. The latter is committed on a local **diagnostic
branch only** (`6ec07978`); neither diagnostic is a production gain. The pinned
libvpx source is `12f3a2ac603e8f10742105519e0cd03c3b8f71dd` (v1.14.1).
Its clean rebuild reproduced the canonical Ducks IVF hash, then the two C
patches instrumented the same source and build options. The instrumented
libvpx binary with search entries is SHA-256
`ff5357bba10b5e72aaf79dd67fea1d155f44134e467cb28246ade76fb92aa2d4`;
the RVP9 entry binary is SHA-256
`d2637a286b0ee911309ae45b3ff7551166481b7457244bbc0a3a5bb90b9246b4`.

Each encoder reproduced **its own** expected IVF SHA-256 on Ducks, Tree and
Town, both before and after adding its search-entry counter. This is diagnostic
output neutrality, not byte parity *between* encoders. `rvp9/` and `libvpx/`
hold phase-only captures; `rvp9-entry/` and `libvpx-entry/` hold the subsequent
entry captures. `summarize.py` verifies manifests, phase rows and entry/phase
consistency and generates `summary.csv` and `validation.json`; run its focused
tests with `python3 -m unittest test_summarize.py` from this directory.

RVP9 records actual production fractional-kernel calls by search family,
shape and phase. Its center `(0,0)` probes are retained in the raw census but
excluded from fractional totals. Libvpx's center probe uses `vf`, not `svf`, so
its `svf` wrapper has no comparable center bucket. Both entry counters count
full-tree search function entries separately; the libvpx SVF wrapper is
otherwise unattributed by caller. These are *descriptive* work comparisons,
not like-for-like timing or evidence that one encoder can adopt the other's
search policy while keeping its output.

The `scripts/` directory pins source/binary/input/output identities and exact
capture commands. Instrumented outputs and CSV are intentionally small; no
large IVF or input corpus is committed. The next step is to attribute Ducks
16×16 entries to caller conditions and look for demonstrably duplicate work,
then screen any bounded exact-output-safe change against native release CPU.

The four `.patch.txt` diffs prefix each line with `|` so unified-diff blank
context lines do not register as trailing whitespace in this docs commit.
To recover an ordinary patch, remove that prefix from every line and change
a line containing only `|` back to one space. The Rust diffs are also retained
in local commits `94aa6025` and `6ec07978` while that diagnostic branch lives.
