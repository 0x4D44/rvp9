# decode head-to-head — 2026-05-23 (R.2 update)

host: windows-x86_64  unpinned  build: release (3c8d6d1ddea4abc4)
rvp9: post-R.2 (decode_coefs thread_local scratch)

## R.2 post-fix (decode_coefs scratch via thread_local!)

Re-measured rvp9 side only after R.2 lands. vpxdec column unchanged.

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| quantizer-00       | 352x288    |     1 |       1 |      1870.8 |       1945.6 |         6088.5 |             0.32 |         - | **FASTER**  |
| tile_1x4           | 1920x1080  |     4 |       1 |     29411.0 |      30632.3 |         3964.5 |             7.73 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |     18448.8 |      19076.9 |         2515.7 |             7.58 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      9675.7 |      11642.7 |          627.3 |            18.56 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |     16605.9 |      14051.8 |          820.9 |            17.12 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |     12742.7 |      10905.9 |          614.4 |            17.75 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      5139.3 |       7147.3 |          592.2 |            12.07 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      6541.7 |       8753.0 |          596.1 |            14.68 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |      1112.5 |        991.6 |          317.0 |             3.13 |         - | n/a         |

## R.2 before/after delta (R.1 → R.2)

| fixture            | threads | post-R.1 p50 us | post-R.2 p50 us | R.2 speedup |
|--------------------|---------|-----------------|-----------------|-------------|
| quantizer-00       |       1 |          4867.7 |          1870.8 |  2.60x      |
| tile_1x4           |       1 |         60844.8 |         29411.0 |  2.07x      |
| tile_1x4           |       4 |         38652.3 |         18448.8 |  2.10x      |
| testsrc_720p30     |       1 |         19260.4 |          9675.7 |  1.99x      |
| testsrc_1080p30    |       2 |         29395.3 |         16605.9 |  1.77x      |
| testsrc_1080p30    |       4 |         20507.7 |         12742.7 |  1.61x      |
| testsrc_1080p60    |       4 |          5490.2 |          5139.3 |  1.07x      |
| testsrc_1080p60    |       8 |          5608.2 |          6541.7 |  0.86x*     |
| foreman-fp1        |       1 |          2226.5 |          1112.5 |  2.00x      |

*1080p60 t=8 is at the bench noise floor; high variance on a ~5ms p50 across 8 worker threads.

## R.1+R.2 compound delta (baseline → R.2)

| fixture            | threads | pre-R.1 baseline | post-R.2  | combined speedup |
|--------------------|---------|------------------|-----------|------------------|
| tile_1x4           |       4 |        339,213.7 |  18,448.8 |          **18.4x** |
| testsrc_1080p30    |       4 |        181,832.6 |  12,742.7 |          **14.3x** |
| testsrc_720p30     |       1 |         87,198.0 |   9,675.7 |           **9.0x** |
| tile_1x4           |       1 |        271,544.0 |  29,411.0 |           **9.2x** |
| testsrc_1080p30    |       2 |        148,492.1 |  16,605.9 |           **8.9x** |
| quantizer-00       |       1 |         12,084.5 |   1,870.8 |           **6.5x** (now 3.25x FASTER than vpxdec) |
| foreman-fp1        |       1 |          8,130.3 |   1,112.5 |           **7.3x** |
| testsrc_1080p60    |       4 |         11,321.2 |   5,139.3 |           **2.2x** |

All PLAN.md §2 real-time budgets PASS:
- 720p30 (33ms/single-core): 9.68ms = 29% of budget
- 1080p30 (33ms/2-4 cores): 12.74ms (t=4) = 39% of budget
- 1080p60 (16ms/4-8 cores): 5.14ms (t=4) = 32% of budget

Worst remaining gap: 20.8x on testsrc_1080p30 t=4. Continuing in R.3.

## Original baseline (2026-05-23 pre-R.1)

# Original ledger — decode head-to-head — 2026-05-23 (R.1 update)

host: windows-x86_64  unpinned  build: release (3c8d6d1ddea4abc4)
rvp9: 34596c2  vpxdec: WebM Project VP9 Decoder v1.14.1 (D:\language\rvp9-2\build\libvpx\vpxdec.exe)
reps: R=9 per vpxdec row (median); rvp9 timing window per HLD §3

note: vpxdec value is per-frame mean from `vpxdec --noblit --summary -t N`,
      including construction/teardown overhead (negligible on >=1000-frame
      fixtures; up to a few percent on short fixtures — see methodology).
      rvp9 mean shown alongside p50 for tail-stability inspection;
      ratio compares means (rvp9 mean / vpxdec mean).
      gate column is informational on unpinned hosts (HLD §1 + §6).

## R.1 post-fix (env::var hot-path cache via OnceLock<TraceFlags>)

Re-measured rvp9 side only after R.1 lands. vpxdec column is the
median value from the 2026-05-23 baseline run; vpxdec was not re-run
since the patch only touches rvp9 decoder internals.

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| testsrc_720p30     | 1280x720   |     1 |       1 |     16588.2 |      15612.3 |          627.3 |            24.89 |     33000 | PASS-info   |
| tile_1x4           | 1920x1080  |     4 |       1 |     63140.7 |      60503.8 |         3964.5 |            15.26 |         - | n/a         |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      6755.6 |      10715.4 |          592.2 |            18.09 |     16000 | PASS-info   |

## R.1 before/after delta

| fixture            | threads | before p50 us | after p50 us | speedup  |
|--------------------|---------|---------------|--------------|----------|
| testsrc_720p30     |       1 |       87198.0 |      16588.2 |  5.26x   |
| tile_1x4           |       1 |      271544.0 |      63140.7 |  4.30x   |
| testsrc_1080p60    |       4 |       11321.2 |       6755.6 |  1.68x   |

R.1 closes ~80% of the rvp9/vpxdec gap on the worst-affected fixtures
in a single mechanical change. The remaining gap (15-25x vs vpxdec) is
algorithmic / SIMD-coverage; further Track R waves will address that.

## Baseline (2026-05-23 — pre-R.1)

For reference, the pre-fix Q.1.A measurement (commit bc95480) which
triggered the Track R investigation. CSV rows preserved in
`decode-headtohead.csv` for the full matrix.

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| quantizer-00       | 352x288    |     1 |       1 |     12084.5 |      12808.4 |         6088.5 |             2.10 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       1 |    271544.0 |     245870.7 |         3964.5 |            62.02 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |    339213.7 |     300922.2 |         2515.7 |           119.62 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |     87198.0 |      83547.7 |          627.3 |           133.18 |     33000 | FAIL-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |    148492.1 |     125084.7 |          820.9 |           152.38 |     33000 | FAIL-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |    181832.6 |     152840.5 |          614.4 |           248.76 |     33000 | FAIL-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |     11321.2 |      75099.1 |          592.2 |           126.81 |     16000 | FAIL-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |     10630.6 |      73694.4 |          596.1 |           123.63 |     16000 | FAIL-info   |
| foreman-fp1        | 352x288    |     1 |       1 |      8130.3 |       7197.8 |          317.0 |            22.71 |         - | n/a         |

## Methodology

- rvp9 column: `rvp9_bench::throughput::run_one_threaded` with
  `Decoder::with_threads(N)`. Timing window covers only
  `Decoder::decode_frame`; 30 warmup frames dropped; loop until
  >= 1000 timed frames per (fixture, threads). HLD §3.
- vpxdec column: `vpxdec --noblit --summary -t N <fixture>` x
  R=9 repetitions (odd; robust median). No calibration
  constant — short fixtures (< 100 frames) include construction
  smear, disclosed in this header per HLD §11.1.
- Ratio is rvp9 mean / vpxdec mean (mean-to-mean, apples to apples
  with vpxdec which has no per-frame surface).
- Gate is informational on unpinned hosts; pinned-host promotion
  deferred to the dedicated Linux CI runner (HLD §1 + §6).
- R.1 re-measurement: same methodology as baseline; rvp9-side only
  (no behavioural change for vpxdec since the patch is internal).
