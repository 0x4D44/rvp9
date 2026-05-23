# decode head-to-head — 2026-05-23T15:54:16Z

host: windows-x86_64  unpinned  build: release (3c8d6d1ddea4abc4)
rvp9: bc95480  vpxdec: WebM Project VP9 Decoder v1.14.1 (D:\language\rvp9-2\build\libvpx\vpxdec.exe)
reps: R=9 per vpxdec row (median); rvp9 timing window per HLD §3

note: vpxdec value is per-frame mean from `vpxdec --noblit --summary -t N`,
      including construction/teardown overhead (negligible on >=1000-frame
      fixtures; up to a few percent on short fixtures — see methodology).
      rvp9 mean shown alongside p50 for tail-stability inspection;
      ratio compares means (rvp9 mean / vpxdec mean).
      gate column is informational on unpinned hosts (HLD §1 + §6).
      rvp9 mean for short fixtures (< MIN_TIMED_FRAMES = 1000) is
      inflated by repeated keyframe re-decode across replay loops
      (steady-state Decoder per HLD §3); p50 is the more stable
      number for those rows.

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
