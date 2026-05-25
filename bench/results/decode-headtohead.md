# decode head-to-head — 2026-05-25T00:26:44Z

host: windows-x86_64  unpinned  build: release (3c8d6d1ddea4abc4)
rvp9: 86aaa92  vpxdec: WebM Project VP9 Decoder v1.14.1 (D:\language\rvp9-2\build\libvpx\vpxdec.exe)
reps: R=3 per vpxdec row (median); rvp9 timing window per HLD §3
rvp9 runs: 1 (summary row = run 1; CSV ledger captures all runs)

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
| quantizer-00       | 352x288    |     1 |       1 |      1700.3 |       1740.1 |         5354.5 |             0.32 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       1 |     15652.3 |      14839.2 |         3045.9 |             4.87 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |      7602.7 |       7518.0 |         2084.0 |             3.61 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      4978.5 |       4502.3 |          552.0 |             8.16 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |      7202.8 |       7004.9 |          760.6 |             9.21 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |      4477.7 |       5236.8 |          520.9 |            10.05 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      3990.8 |       4331.0 |          559.6 |             7.74 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      4228.6 |       4647.7 |          529.2 |             8.78 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       662.3 |        642.4 |          283.9 |             2.26 |         - | n/a         |

## Methodology

- rvp9 column: `rvp9_bench::throughput::run_one_threaded` with
  `Decoder::with_threads(N)`. Timing window covers only
  `Decoder::decode_frame`; 30 warmup frames dropped; loop until
  >= 1000 timed frames per (fixture, threads). HLD §3.
- vpxdec column: `vpxdec --noblit --summary -t N <fixture>` x
  R=3 repetitions (odd; robust median). No calibration
  constant — short fixtures (< 100 frames) include construction
  smear, disclosed in this header per HLD §11.1.
- Ratio is rvp9 mean / vpxdec mean (mean-to-mean, apples to apples
  with vpxdec which has no per-frame surface).
- Gate is informational on unpinned hosts; pinned-host promotion
  deferred to the dedicated Linux CI runner (HLD §1 + §6).
