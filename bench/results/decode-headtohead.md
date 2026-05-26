# decode head-to-head — 2026-05-26T06:24:49Z

host: windows-x86_64  unpinned  build: release (3c8d6d1ddea4abc4)
rvp9: 1993974  vpxdec: WebM Project VP9 Decoder v1.14.1 (D:\language\rvp9-2\build\libvpx\vpxdec.exe)
reps: R=5 per vpxdec row (median); rvp9 timing window per HLD §3
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
| quantizer-00       | 352x288    |     1 |       1 |      1295.3 |       1331.6 |         5169.0 |             0.26 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       1 |     10633.6 |      10277.5 |         3200.3 |             3.21 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |      4022.3 |       3962.8 |         1966.2 |             2.02 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      3511.2 |       3168.6 |          533.6 |             5.94 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |      4776.3 |       4198.7 |          682.1 |             6.16 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |      3024.1 |       2708.8 |          565.7 |             4.79 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      1615.3 |       1995.5 |          452.6 |             4.41 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      1621.5 |       1995.5 |          462.2 |             4.32 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       499.4 |        526.0 |          301.0 |             1.75 |         - | n/a         |

## Methodology

- rvp9 column: `rvp9_bench::throughput::run_one_threaded` with
  `Decoder::with_threads(N)`. Timing window covers only
  `Decoder::decode_frame`; 30 warmup frames dropped; loop until
  >= 1000 timed frames per (fixture, threads). HLD §3.
- vpxdec column: `vpxdec --noblit --summary -t N <fixture>` x
  R=5 repetitions (odd; robust median). No calibration
  constant — short fixtures (< 100 frames) include construction
  smear, disclosed in this header per HLD §11.1.
- Ratio is rvp9 mean / vpxdec mean (mean-to-mean, apples to apples
  with vpxdec which has no per-frame surface).
- Gate is informational on unpinned hosts; pinned-host promotion
  deferred to the dedicated Linux CI runner (HLD §1 + §6).
