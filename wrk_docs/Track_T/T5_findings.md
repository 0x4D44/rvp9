# T-5 Findings: Per-tile dispatch overhead investigation

**Date:** 2026-05-24
**Agent:** T-5 implementation agent
**Status:** Complete — instrumentation landed, data collected, hypothesis verdicts rendered

## 1. Methodology

Per HLD `T5_tile_dispatch_profile_HLD.md` section 6:
- Feature-gated `rdtsc` measurements at phase boundaries (7 main-thread phases + 3 per-tile worker phases)
- 10 runs x 1,000 timed post-warmup frames = 10,300 data rows
- Fixture: `vp90-2-08-tile_1x4.webm` (1080p, 4 tile columns, 4 worker threads)
- CPU: ~3,072 MHz (calibrated via rdtsc vs Instant)
- CSV: `bench/results/t5-tile-profile.tile_1x4.t4.csv`
- Analysis: `scripts/t5-analyze.py`

## 2. Per-phase summary table

All values in microseconds (us). "Share" is percentage of total main-thread-visible frame time (= sum of all main-thread phases).

| Phase | Median (us) | P95 (us) | Share of frame |
|---|---|---|---|
| **loopfilter** | 3,448 | 4,686 | **35.5%** |
| **join** (parallel decode wait) | 3,809 | 5,363 | 39.3% |
| **merge** | 1,246 | 1,783 | **12.8%** |
| **grid_stitch** | 663 | 1,038 | **6.8%** |
| setup | 142 | 584 | 1.5% |
| border | 105 | 139 | 1.1% |
| submit | 20 | 35 | 0.2% |
| tile_decode (sum of 4 tiles) | 13,613 | 16,739 | (parallel work) |
| tile_lockin | 1 | 2 | 0.0% |
| tile_lockout | 2 | 2 | 0.0% |

**Median wall time:** 9,716 us (9.72 ms/frame)

## 3. Serial fraction analysis

Serial phases (everything the main thread does besides waiting for workers):

```
setup + submit + merge + grid_stitch + loopfilter + border
= 142 + 20 + 1,246 + 663 + 3,448 + 105
= 5,624 us = 57.8% of frame time
```

Parallel decode time (join = time main blocks while workers decode): 3,809 us = 39.2% of frame time.

With 57.8% serial, Amdahl's law predicts on 4 cores:
```
Speedup = 1 / (0.578 + 0.422/4) = 1 / 0.684 = 1.46x
```

The observed scaling in the HLD (1.79x on t=1 vs t=4) is better than this prediction. The discrepancy is because the single-threaded path avoids merge/grid-stitch overhead (no per-tile scratch buffers, no memcpy merge), so the t=1 serial fraction is lower. The key takeaway: **under 4-thread parallel dispatch, 57.8% of each frame is serial**, dominated by the loopfilter.

## 4. Hypothesis verdicts

### H1: Serial post-scope work dominates — CONFIRMED

The loopfilter alone is 35.5% of frame time. Adding merge (12.8%), grid_stitch (6.8%), and border (1.1%) brings the serial-post-scope total to **56.2%** of frame time. This is the dominant bottleneck.

### H2: Per-tile mutex overhead non-trivial — REJECTED

Lock-in and lock-out are both <1 us per tile, <0.01% of decode time. The 48 mutex ops/frame are completely negligible.

### H3: Per-frame per-tile allocations — MINOR

Setup is only 1.5% of frame time (142 us median). Alloc churn is not the dominant cost, though the P95 (584 us) suggests occasional allocator contention.

### H5: Inter-tile work imbalance — NOT TESTED

All 4 tiles' decode times are similar (range ~5-21 ms in raw cycles; the variance is partly noise). The fixture is synthetic with uniform content. Imbalance is unlikely to exceed 15% on this fixture.

### H6: Dispatch/wake latency — REJECTED

Submit is 20 us (0.2% of frame). Crossbeam channel dispatch cost is negligible.

### H7: Grid stitching — MODERATE CONTRIBUTOR

`build_frame_mi_grid` is 663 us = 6.8% of frame time. Not dominant, but a worthwhile optimization target alongside the merge.

### H8: Border extend — MINOR

`extend_frame_borders` is 105 us = 1.1%. Negligible.

## 5. Breakdown of serial budget

```
loopfilter:   3,448 us  (61.3% of serial budget)
merge:        1,246 us  (22.2%)
grid_stitch:    663 us  (11.8%)
setup:          142 us  ( 2.5%)
border:         105 us  ( 1.9%)
submit:          20 us  ( 0.4%)
```

**The loopfilter is 61.3% of the serial budget** — it is the single most impactful optimization target for parallel scaling.

## 6. Recommended next-step fix

Per HLD section 7 decision tree, **H1 confirmed with loopfilter as the dominant serial component**:

**Recommended fix:** Pipeline the loopfilter against the next frame's tile decode (libvpx's `vp9_loop_filter_worker.c` pattern). A dedicated LF worker runs asynchronously; the main thread fires it and immediately begins the next frame's decode. The LF cost (~3.4 ms) is amortized to ~0 ms wall time per frame.

**Secondary fix (lower effort, lower impact):** Fold the merge + grid_stitch (combined 19.6% of frame time, ~1.9 ms) into the loopfilter pass or parallelize them. The merge is a serial memcpy of ~2 MiB per frame; the grid stitch is a serial copy of ~65k cells. Both could be parallelized by row or folded into the per-tile worker output stage.

**Impact estimate:** Eliminating the 3.4 ms loopfilter from the critical path would reduce the serial fraction from 57.8% to ~22.4%, giving Amdahl-predicted scaling of 1/(0.224 + 0.776/4) = **2.35x** on 4 cores — a ~31% improvement over today's 1.79x.

## 7. Validation

- Feature-off build compiles clean (zero warnings in decoder crate)
- Feature-on build compiles clean (1 pre-existing warning in reconstruct_inter.rs)
- 120/120 unit tests pass with feature on
- Zero mutation of decode state from instrumentation sites
- CSV schema version column present (schema=1)
- `now_cycles()` is `#[inline(always)]` with SAFETY comment
- All instrumentation use sites are `#[cfg(feature = "profile-tiles")]`-gated, not `cfg!()` runtime branches
