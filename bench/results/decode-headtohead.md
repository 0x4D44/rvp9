# decode head-to-head — 2026-05-25T07:29:52Z

host: windows-x86_64  unpinned  build: release (3c8d6d1ddea4abc4)
rvp9: 1fb62af  vpxdec: WebM Project VP9 Decoder v1.14.1 (D:\language\rvp9-2\.claude\worktrees\agent-a7293cd19c26fac9c\build\libvpx\vpxdec.exe)
reps: R=3 per vpxdec row (median); rvp9 timing window per HLD §3
rvp9 runs: 3 (summary row = run 1; CSV ledger captures all runs)

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
| quantizer-00       | 352x288    |     1 |       1 |      1538.9 |       1585.4 |                |                ∞ |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       1 |     14597.1 |      13774.5 |                |                ∞ |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |      7252.2 |       7137.1 |                |                ∞ |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      4595.0 |       4128.0 |                |                ∞ |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |      6191.0 |       6022.0 |                |                ∞ |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |      3716.9 |       4406.8 |                |                ∞ |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      3553.1 |       3897.8 |                |                ∞ |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      3361.2 |       3741.5 |                |                ∞ |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       616.7 |        597.1 |                |                ∞ |         - | n/a         |

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
- Multi-run sweep: -Runs 3. The summary table above shows
  run 1 only; the CSV ledger contains all 3 rows per
  fixture×threads (same 	imestamp, un_index=1..3).
  Pair with `bench-compare --baseline <ts1> --candidate <ts2>`
  for Mann-Whitney U significance against a previous sweep.
- vpxdec sweep skipped (-SkipVpxdec); vpxdec_* and
  ratio_mean columns blank in the CSV for this timestamp.
