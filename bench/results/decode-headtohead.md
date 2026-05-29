# decode head-to-head — 2026-05-29T21:37:13Z

host: windows-x86_64  unpinned  build: release (3c8d6d1ddea4abc4)
rvp9: 07c70e4  vpxdec: WebM Project VP9 Decoder v1.14.1 (D:\language\rvp9-2\build\libvpx\vpxdec.exe)
reps: R=9 per vpxdec row (median); rvp9 timing window per HLD §3
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
| quantizer-00       | 352x288    |     1 |       1 |      1685.0 |       1729.8 |         6040.0 |             0.29 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       1 |     15705.9 |      14895.4 |         4374.6 |             3.40 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |      4411.6 |       4407.4 |         2973.7 |             1.48 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      4976.4 |       4472.4 |          663.0 |             6.75 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |      6214.3 |       5196.5 |         1129.5 |             4.60 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |      3417.8 |       3093.7 |          720.0 |             4.30 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      2455.0 |       2588.4 |          677.7 |             3.82 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      2868.9 |       2917.5 |          685.1 |             4.26 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       711.3 |        665.5 |          332.9 |             2.00 |         - | n/a         |

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
