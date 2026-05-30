# convolve8 AVX2 vs libvpx — baseline + analysis (2026-05-30)

## Baseline (run-kernel-audit.ps1, min-of-5, pinned, bit-exact InMatch/OutMatch=True)
| kernel | size | rvp9 ns | vpx ns | ratio |
|---|---|---|---|---|
| convolve8_2d    | 8x8   | 25.06  | 16.46  | **1.52** |
| convolve8_2d    | 16x16 | 49.85  | 38.48  | 1.30 |
| convolve8_2d    | 32x32 | 136.00 | 125.80 | 1.08 |
| convolve8_2d    | 64x64 | 481.65 | 432.60 | 1.11 |
| convolve8_horiz | 16x16 | 21.73  | 16.56  | 1.31 |
| convolve8_horiz | 64x64 | 262.50 | 219.33 | 1.20 |
| convolve8_vert  | 16x16 | 21.83  | 15.21  | **1.44** |
| convolve8_vert  | 64x64 | 194.17 | 186.73 | 1.04 |

## KEY FINDING: the gap is SMALL-SIZE FIXED OVERHEAD, not throughput
Large sizes (32/64) are near-parity (1.04–1.20×); small sizes (8x8/16x16) are 1.3–1.52×.
That is a per-call FIXED-overhead signature (amortized away at large sizes). The worst case
is **8x8 2-D = 1.52×, which is THE most common inter block** — and at w=8 the libvpx-style
16-wide horizontal grouping does NOT apply, so the static "16-wide horiz" lever (below) misses
the common case. The vertical kernel is NOT at parity at 16x16 (1.44×) either — also fixed-cost.

For real video (foreman = mostly 8x8/16x16 inter MC), the highest-value lever is
**reducing per-call fixed overhead** on the small-block path of BOTH single-axis kernels.

## Static analysis (sub-agent, evidence-based) — the throughput lever (helps LARGE horiz only)
Horizontal kernel produces 8 cols × 2 rows per maddubs group (2× 8-byte storel), so for w>=16
the inner loop runs twice with a per-8-col packus/cast/extracti128/storel epilogue. libvpx does
16 cols × 1 row per group (1× 16-byte store). ~40 vs ~26 vector ops for 16x16 horiz.
Ranked plan (all byte-exact; oracle = differential_all_filters in the kernel files):
1. 16-wide horizontal path (w>=16): one 16-col group/row, store 16B. Helps horiz 16/32/64 + 2-D
   pass. The vertical kernel is already a faithful libvpx port for LARGE sizes (don't touch).
2. 4-row unroll for the w=8 horizontal pair loop — helps 8x8 (most common).
3. Hoist horiz shuffle masks (m01..m67) to module-level aligned constants (free).
4. Last-row scratch elision for the internal 2-D path (the padded intermediate makes the
   over-read safe) — removes a per-call zeroed [u8;80] + copy + scalarized last-row.

## Reconciled priority (audit + analysis)
The common-case win (8x8/16x16) is FIXED-OVERHEAD: items 2 + 4 (+ investigate the per-call tap
broadcast and the vert small-size setup). Item 1 (16-wide horiz) helps the rarer large blocks.
Convolve8 already got the 2-D memset-elision win this session (committed 2e6e0cf).

## Status: ANALYSIS + BASELINE done; implementation is a focused next effort
The AVX2 kernel rewrite is byte-exact-critical (maddubs operands + (x0+x2)+(x1+x3)+64 / sat-add
/ srai7 / packus reduction must stay identical; only data layout changes). Iterate with the FAST
oracle (the differential_* unit tests, seconds) then gate on conformance 6/6.
</content>
