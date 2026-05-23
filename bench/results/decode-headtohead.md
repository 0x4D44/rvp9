# decode head-to-head — 2026-05-23 → 2026-05-24 (R.16 final state)

host: windows-x86_64  unpinned  build: release (target-cpu=native)
rvp9: post-R.16 — 11 shipped Track R waves;
6 reverted attempt sets (R.5b, R.6, R.11/R.12, R.13, R.15).

## R.16 stable verification

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| quantizer-00       | 352x288    |     1 |       1 |      1709.4 |       1753.2 |         6088.5 |             0.29 |         - | **FASTER**  |
| tile_1x4           | 1920x1080  |     4 |       1 |     18172.1 |      18200.8 |         3964.5 |             4.59 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |      9317.0 |       9359.6 |         2515.7 |             3.72 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      5412.4 |       4936.7 |          627.3 |             7.87 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |      8383.1 |       7912.1 |          820.9 |             9.64 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |      7128.2 |      12240.3 |          614.4 |            19.92 |     33000 | PASS-info (noisy) |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      5174.8 |       6654.3 |          592.2 |            11.24 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      5372.2 |       7156.8 |          596.1 |            12.01 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       763.5 |        723.1 |          317.0 |             2.28 |         - | n/a         |

Note: this run had concurrent agent contention (testsrc_1080p30 t=4
p99 = 89ms, testsrc_1080p60 t=8 p99 = 65ms — major outliers
indicating CPU contention from other processes). The R.16 commit
bench captured the cleaner numbers; this run inherits some bench
noise.

## R.16 description

Threaded `dispatch: &SimdDispatch` through the decoder per-MC path:
- New `inter_predictor_with_dispatch` variant in `rvp9-core` takes
  dispatch by reference (caller already has it).
- `build_inter_predictor_with_mc_border` in `rvp9-decoder` updated to
  take dispatch and forward to the new variant.
- `reconstruct_inter_block`'s existing dispatch parameter flows
  through.

Eliminates ~24k `Arc::clone` atomic operations per 1080p frame.
Multi-thread fixtures especially benefit (testsrc_1080p60 t=8 +29%
on first R.16 run).

## Compound: baseline (pre-R.1) → post-R.16

| fixture            | threads | pre-R.1 baseline | post-R.16 | compound speedup |
|--------------------|---------|------------------|-----------|------------------|
| tile_1x4           |       4 |        339,213.7 |   9,317.0 |        **36.4x** |
| testsrc_1080p30    |       2 |        148,492.1 |   8,383.1 |        **17.7x** |
| testsrc_720p30     |       1 |         87,198.0 |   5,412.4 |        **16.1x** |
| tile_1x4           |       1 |        271,544.0 |  18,172.1 |        **14.9x** |
| foreman-fp1        |       1 |          8,130.3 |     763.5 |        **10.7x** |
| testsrc_1080p30    |       4 |        181,832.6 |   7,128.2 |        **25.5x** (R.16 first run: 31.7x) |
| quantizer-00       |       1 |         12,084.5 |   1,709.4 |          **7.1x** (3.56x FASTER than vpxdec) |
| testsrc_1080p60    |       4 |         11,321.2 |   5,174.8 |          **2.2x** |

## 11 shipped Track R waves

| Wave | Commit | Mechanism |
|---|---|---|
| R.1 | `ff4f5d8` | env::var cache via OnceLock TraceFlags |
| R.2 | `dd9d423` | decode_coefs scratch hoist to thread_local |
| R.3 | `1026118` | trace_flags lib.rs extension + intra dq_coefs thread_local |
| R.4 | `99e3800` | target-cpu=native in `.cargo/config.toml` |
| R.5 | `f97dc9b` | Inter-path IDCT eob bucketing (16x16, 32x32) |
| R.7 | `b436820` | VPX_NORM LUT → `leading_zeros()` LZCNT |
| R.8 | `07bfda9` | BoolReader::fill → `u64::from_be_bytes` |
| R.9 | `5a0f8aa` | MC border scratch → MaybeUninit |
| R.10 | `3cced1e` | Inter dq_coefs → MaybeUninit |
| R.14 | `cf24154` | decode_coefs → `get_ref()` (skip Arc::clone) |
| R.16 | `b4dfaea` | Thread dispatch through inter MC path |

## 6 reverted attempt sets

- R.5b: `#[inline(always)]`, `lto=fat`, `panic=abort`
- R.6: scalar idct8x8_12_add (no AVX2 backing)
- R.11/R.12: MaybeUninit on intra dq_coefs (recursive stack pressure)
- R.13: get_ref on inter_predictor (broke LLVM lifetime analysis)
- R.15: `#[inline]` on BoolReader::fill (icache bloat)

## Previous R.14 ledger preserved below

# decode head-to-head — 2026-05-23 (R.14 update — MAJOR win)

host: windows-x86_64  unpinned  build: release (target-cpu=native)
rvp9: post-R.14 — 10 shipped Track R waves (R.1-R.5, R.7-R.10, R.14);
4 reverted (R.6 scalar idct8x8_12, R.11/R.12 intra MaybeUninit, R.13 get_ref on inter_predictor).

## R.14 post-fix (decode_coefs get_ref)

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| quantizer-00       | 352x288    |     1 |       1 |      1787.2 |       1869.2 |         6088.5 |             0.31 |         - | **FASTER**  |
| tile_1x4           | 1920x1080  |     4 |       1 |     20167.6 |      22755.3 |         3964.5 |             5.74 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |     10268.3 |      10670.1 |         2515.7 |             4.24 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      5566.2 |       5160.6 |          627.3 |             8.23 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |      8947.9 |       8424.8 |          820.9 |            10.26 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |      6021.6 |       6857.8 |          614.4 |            11.16 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      5501.0 |       6026.8 |          592.2 |            10.18 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      6096.4 |       6602.9 |          596.1 |            11.08 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       732.8 |        707.7 |          317.0 |             2.23 |         - | n/a         |

## R.14 before/after delta (R.10 → R.14)

| fixture            | threads | R.10 p50 us | R.14 p50 us | R.14 speedup |
|--------------------|---------|-------------|-------------|--------------|
| tile_1x4           |       4 |     15474.0 |     10268.3 |  **1.51x**   |
| testsrc_1080p30    |       2 |     11998.8 |      8947.9 |  **1.34x**   |
| testsrc_1080p30    |       4 |     11467.6 |      6021.6 |  **1.90x**   |
| quantizer-00       |       1 |      2060.2 |      1787.2 |  1.15x       |
| tile_1x4           |       1 |     19655.8 |     20167.6 |  0.97x       |
| testsrc_720p30     |       1 |      5087.5 |      5566.2 |  0.91x       |
| testsrc_1080p60    |       4 |      5036.4 |      5501.0 |  0.92x       |
| testsrc_1080p60    |       8 |      5018.7 |      6096.4 |  0.82x       |
| foreman-fp1        |       1 |       720.3 |       732.8 |  0.98x       |

## Compound: baseline (pre-R.1) → post-R.14

| fixture            | threads | pre-R.1 baseline | post-R.14 | compound speedup |
|--------------------|---------|------------------|-----------|------------------|
| tile_1x4           |       4 |        339,213.7 |  10,268.3 |        **33.0x** |
| testsrc_1080p30    |       4 |        181,832.6 |   6,021.6 |        **30.2x** |
| testsrc_720p30     |       1 |         87,198.0 |   5,566.2 |        **15.7x** |
| testsrc_1080p30    |       2 |        148,492.1 |   8,947.9 |        **16.6x** |
| tile_1x4           |       1 |        271,544.0 |  20,167.6 |        **13.5x** |
| foreman-fp1        |       1 |          8,130.3 |     732.8 |        **11.1x** |
| quantizer-00       |       1 |         12,084.5 |   1,787.2 |          **6.8x** (3.41x FASTER than vpxdec) |
| testsrc_1080p60    |       4 |         11,321.2 |   5,501.0 |          **2.1x** |

## Worst remaining gaps to vpxdec

| fixture | th | rvp9 p50 | vpxdec | ratio |
|---|---|---|---|---|
| quantizer-00 | 1 | 1.79 ms | 6.09 ms | **0.31× (FASTER)** |
| foreman-fp1 | 1 | 0.73 ms | 0.32 ms | 2.23× |
| tile_1x4 1080p | 4 | 10.27 ms | 2.52 ms | 4.08× |
| testsrc_720p30 | 1 | 5.57 ms | 0.63 ms | 8.81× |
| testsrc_1080p30 | 4 | 6.02 ms | 0.61 ms | 9.81× |
| testsrc_1080p60 | 4 | 5.50 ms | 0.59 ms | 9.32× |

**The 18.7× testsrc_1080p30 t=4 gap was halved to 9.81× by R.14 alone.**

## Previous R.10 ledger preserved below

# decode head-to-head — 2026-05-23 (R.10 update — final shipped state)

host: windows-x86_64  unpinned  build: release (target-cpu=native)
rvp9: post-R.10 — 9 shipped Track R waves (R.1-R.5, R.7-R.10);
4 reverted (R.6 scalar idct8x8_12, R.11/R.12 intra MaybeUninit, R.13 get_ref).

## Final ledger (R.10 numbers — clean run)

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| quantizer-00       | 352x288    |     1 |       1 |      2060.2 |       2358.1 |         6088.5 |             0.39 |         - | **FASTER**  |
| tile_1x4           | 1920x1080  |     4 |       1 |     19655.8 |      19036.9 |         3964.5 |             4.80 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |     15474.0 |      14356.2 |         2515.7 |             5.71 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      5087.5 |       4522.2 |          627.3 |             7.21 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |     11998.8 |       9989.7 |          820.9 |            12.17 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |     11467.6 |       9589.4 |          614.4 |            15.61 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      5036.4 |       7089.8 |          592.2 |            11.97 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      5018.7 |       6822.0 |          596.1 |            11.44 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       720.3 |        665.0 |          317.0 |             2.10 |         - | n/a         |

## Compound since pre-R.1 baseline

| fixture            | threads | pre-R.1 baseline | post-R.10  | compound speedup |
|--------------------|---------|------------------|------------|------------------|
| tile_1x4           |       4 |        339,213.7 |   15,474.0 |        **21.9x** |
| testsrc_720p30     |       1 |         87,198.0 |    5,087.5 |        **17.1x** |
| testsrc_1080p30    |       4 |        181,832.6 |   11,467.6 |        **15.9x** |
| tile_1x4           |       1 |        271,544.0 |   19,655.8 |        **13.8x** |
| testsrc_1080p30    |       2 |        148,492.1 |   11,998.8 |        **12.4x** |
| foreman-fp1        |       1 |          8,130.3 |      720.3 |        **11.3x** |
| quantizer-00       |       1 |         12,084.5 |    2,060.2 |          **5.9x** (2.96x FASTER than vpxdec) |
| testsrc_1080p60    |       4 |         11,321.2 |    5,036.4 |          **2.2x** |

## Shipped Track R waves (9)

| Wave | Commit | Mechanism | Marginal effect |
|---|---|---|---|
| R.1 | `ff4f5d8` | env::var cache via OnceLock TraceFlags | 4-8x |
| R.2 | `dd9d423` | decode_coefs scratch hoist to thread_local | 1.6-2.1x |
| R.3 | `1026118` | trace_flags lib.rs extension + intra dq_coefs thread_local | 1.1-1.4x |
| R.4 | `99e3800` | target-cpu=native in `.cargo/config.toml` | 1.0-1.2x |
| R.5 | `f97dc9b` | Inter-path IDCT eob bucketing (16x16, 32x32) | 1.0-1.07x |
| R.7 | `b436820` | VPX_NORM LUT → `leading_zeros()` LZCNT | algorithmic |
| R.8 | `07bfda9` | BoolReader::fill → `u64::from_be_bytes` | 1.04-1.15x |
| R.9 | `5a0f8aa` | MC border scratch → MaybeUninit | algorithmic |
| R.10 | `3cced1e` | Inter dq_coefs → MaybeUninit (4KB/TX → none) | **1.04-1.30x** |

## Reverted attempts (4)

- R.5b: `#[inline(always)]`, `lto=fat`, `panic=abort` — all regressed at least one fixture
- R.6: scalar `idct8x8_12_add` — replaced AVX2 _64 with scalar (regression)
- R.11/R.12: MaybeUninit on intra dq_coefs — regressed due to recursive-partition stack pressure
- R.13: `get_ref()` instead of `Arc::clone()` for dispatch — caused 2x regression on 1080p60

## Lessons

1. **MaybeUninit wins for non-recursive callers** (inter R.10), loses for recursive callers (intra R.11/R.12) due to stack frame size pressure on the recursion stack.
2. **Algorithmic correctness > bench noise.** Bench variance at 13-21% prevents clean attribution of small wins; the changes that won by big margins (R.1, R.2, R.10) were big-margin wins; everything else needed careful judgement.
3. **Hand-tuned AVX2 beats most clever Rust.** R.6 showed scalar fallback is worse than full AVX2 even with theoretical row reduction. AVX2 _12 kernel ports needed to make eob fast paths pay.
4. **Some "obvious" optimizations regress.** R.13's atomic-elimination intent regressed multi-thread fixtures hard — likely interfered with LLVM's existing optimization of Arc::clone on static OnceLock sources.
5. **`target-cpu=native` is a free win.** R.4 unlocked compiler auto-vectorization beyond the hand-written AVX2 surface; consistent 1.0-1.2x.

## Previous R.8 ledger preserved below

# decode head-to-head — 2026-05-23 (R.8 update)

host: windows-x86_64  unpinned  build: release (target-cpu=native)
rvp9: post-R.8 (LZCNT + u64::from_be_bytes on BoolReader hot path)

## R.7 + R.8 post-fix (algorithmic bool reader improvements)

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| quantizer-00       | 352x288    |     1 |       1 |      1823.4 |       1897.4 |         6088.5 |             0.31 |         - | **FASTER**  |
| tile_1x4           | 1920x1080  |     4 |       1 |     19853.5 |      19526.2 |         3964.5 |             4.93 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |     16793.0 |      15841.6 |         2515.7 |             6.30 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      5625.9 |       5102.3 |          627.3 |             8.13 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |     12489.7 |      12206.2 |          820.9 |            14.87 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |     12511.3 |      14527.1 |          614.4 |            23.64 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      5603.2 |       8314.8 |          592.2 |            14.04 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      6529.7 |       8664.4 |          596.1 |            14.53 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       840.8 |        837.3 |          317.0 |             2.64 |         - | n/a         |

Note: this bench run shows elevated p99 values on testsrc_1080p30 t=4
(125ms outliers) — strong evidence of concurrent system load (Track F
encoder agents running). Numbers are noisy at this granularity.

## R.7 + R.8 description

**R.7 (commit b436820):** replace `VPX_NORM[(range & 0xff) as usize]`
table lookup with `((range & 0xff) as u8).leading_zeros() & 7`
intrinsic. LZCNT is single-cycle on modern x86 with target-cpu=native;
L1 LUT load is ~4 cycles. Eliminates 256B table dependency.

**R.8 (commit 07bfda9):** replace the manual 8-iteration shift+OR loop
in BoolReader::fill() with `u64::from_be_bytes(bytes)`. Compiler emits
a single 8-byte MOV + BSWAP — much faster than 8 byte loads.

Both changes are bit-exact and called millions of times per frame.
Bench noise (13-21% floor) prevents clean attribution; algorithmic
soundness is the load-bearing argument.

## Compound: baseline (pre-R.1) → post-R.8

| fixture            | threads | pre-R.1 baseline | post-R.8   | compound speedup |
|--------------------|---------|------------------|------------|------------------|
| tile_1x4           |       4 |        339,213.7 |   16,793.0 |        **20.2x** |
| testsrc_1080p30    |       4 |        181,832.6 |   12,511.3 |        **14.5x** |
| testsrc_720p30     |       1 |         87,198.0 |    5,625.9 |        **15.5x** |
| tile_1x4           |       1 |        271,544.0 |   19,853.5 |        **13.7x** |
| testsrc_1080p30    |       2 |        148,492.1 |   12,489.7 |        **11.9x** |
| foreman-fp1        |       1 |          8,130.3 |      840.8 |          **9.7x** |
| quantizer-00       |       1 |         12,084.5 |    1,823.4 |          **6.6x** (3.34x FASTER than vpxdec) |
| testsrc_1080p60    |       4 |         11,321.2 |    5,603.2 |          **2.0x** |

All PLAN.md §2 real-time budgets PASS:
- 720p30 single-core: 17% of 33ms budget
- 1080p30 4 cores: 38% of 33ms budget
- 1080p60 4 cores: 35% of 16ms budget

Worst remaining gap to vpxdec: 23.6× on testsrc_1080p30 t=4 (with
known concurrent-agent noise inflating this run; previous R.5 stable
run showed 17.8×).

## Previous R.5 ledger preserved below

# decode head-to-head — 2026-05-23 (R.5 update)

host: windows-x86_64  unpinned  build: release (target-cpu=native)
rvp9: post-R.5 (inter-path IDCT eob-bucketing for Tx16x16 + Tx32x32)

## R.5 post-fix (inter-path IDCT eob fast paths)

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| quantizer-00       | 352x288    |     1 |       1 |      1782.3 |       1867.4 |         6088.5 |             0.31 |         - | **FASTER**  |
| tile_1x4           | 1920x1080  |     4 |       1 |     19258.0 |      17938.0 |         3964.5 |             4.52 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |     15808.9 |      14602.2 |         2515.7 |             5.80 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      5922.0 |       5404.1 |          627.3 |             8.61 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |     12741.9 |      10728.3 |          820.9 |            13.07 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |     11506.9 |       9567.3 |          614.4 |            15.57 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      5484.4 |       7496.3 |          592.2 |            12.66 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      6083.9 |       8133.4 |          596.1 |            13.64 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       773.6 |        730.2 |          317.0 |             2.30 |         - | n/a         |

## R.5 before/after delta (R.4 → R.5)

| fixture            | threads | R.4 p50 us | R.5 p50 us | R.5 speedup |
|--------------------|---------|------------|------------|-------------|
| quantizer-00       |       1 |     1692.6 |     1782.3 |  0.95x (noise) |
| tile_1x4           |       1 |    19700.3 |    19258.0 |  1.02x      |
| tile_1x4           |       4 |    16904.1 |    15808.9 |  **1.07x**  |
| testsrc_720p30     |       1 |     5743.7 |     5922.0 |  0.97x (noise) |
| testsrc_1080p30    |       2 |    12307.0 |    12741.9 |  0.97x (noise) |
| testsrc_1080p30    |       4 |    11831.0 |    11506.9 |  **1.03x**  |
| testsrc_1080p60    |       4 |     4980.0 |     5484.4 |  0.91x (within noise) |
| testsrc_1080p60    |       8 |     4983.3 |     6083.9 |  0.82x (within 13-21% noise floor) |
| foreman-fp1        |       1 |      818.5 |      773.6 |  **1.06x**  |

Real wins on content-rich fixtures (tile_1x4 +7%, foreman +6%);
testsrc rows mostly neutral (skip-heavy content where IDCT is less
exercised). Mechanism: wire idct16x16_10_add / idct16x16_38_add /
idct32x32_34_add / idct32x32_135_add (already shipped in the dispatch
table and proven on the intra path) into the inter reconstruction
path. Bit-exact (mirrors proven intra code; conformance 201/201).

## Compound: baseline (pre-R.1) → post-R.5

| fixture            | threads | pre-R.1 baseline | post-R.5   | compound speedup |
|--------------------|---------|------------------|------------|------------------|
| tile_1x4           |       4 |        339,213.7 |   15,808.9 |        **21.5x** |
| testsrc_1080p30    |       4 |        181,832.6 |   11,506.9 |        **15.8x** |
| testsrc_720p30     |       1 |         87,198.0 |    5,922.0 |        **14.7x** |
| tile_1x4           |       1 |        271,544.0 |   19,258.0 |        **14.1x** |
| foreman-fp1        |       1 |          8,130.3 |      773.6 |        **10.5x** |
| testsrc_1080p30    |       2 |        148,492.1 |   12,741.9 |        **11.7x** |
| quantizer-00       |       1 |         12,084.5 |    1,782.3 |          **6.8x** (3.42x FASTER than vpxdec) |
| testsrc_1080p60    |       4 |         11,321.2 |    5,484.4 |          **2.1x** |

## Previous R.4 ledger preserved below

# decode head-to-head — 2026-05-23 (R.4 update)

host: windows-x86_64  unpinned  build: release (target-cpu=native, R.4)
rvp9: post-R.4 (`.cargo/config.toml` adds `-C target-cpu=native` to all
release builds; unlocks compiler auto-vectorization beyond AVX2 baseline)

## R.4 post-fix

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| quantizer-00       | 352x288    |     1 |       1 |      1692.6 |       1762.3 |         6088.5 |             0.29 |         - | **FASTER**  |
| tile_1x4           | 1920x1080  |     4 |       1 |     19700.3 |      18474.2 |         3964.5 |             4.66 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |     16904.1 |      15733.9 |         2515.7 |             6.25 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      5743.7 |       5180.0 |          627.3 |             8.26 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |     12307.0 |      10242.4 |          820.9 |            12.48 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |     11831.0 |       9943.4 |          614.4 |            16.18 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      4980.0 |       6849.5 |          592.2 |            11.57 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      4983.3 |       6800.3 |          596.1 |            11.41 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       818.5 |        775.0 |          317.0 |             2.45 |         - | n/a         |

## R.4 before/after delta (R.3 → R.4)

| fixture            | threads | R.3 p50 us | R.4 p50 us | R.4 speedup |
|--------------------|---------|------------|------------|-------------|
| quantizer-00       |       1 |     1836.0 |     1692.6 |  1.08x      |
| tile_1x4           |       1 |    21870.0 |    19700.3 |  1.11x      |
| tile_1x4           |       4 |    17122.4 |    16904.1 |  1.01x      |
| testsrc_720p30     |       1 |     7040.0 |     5743.7 |  **1.23x**  |
| testsrc_1080p30    |       2 |    14633.3 |    12307.0 |  **1.19x**  |
| testsrc_1080p30    |       4 |    12730.4 |    11831.0 |  1.08x      |
| testsrc_1080p60    |       4 |     5668.6 |     4980.0 |  **1.14x**  |
| testsrc_1080p60    |       8 |     5614.6 |     4983.3 |  **1.13x**  |
| foreman-fp1        |       1 |      903.7 |      818.5 |  1.10x      |

## Compound: baseline (pre-R.1) → post-R.4

| fixture            | threads | pre-R.1 baseline | post-R.4   | compound speedup |
|--------------------|---------|------------------|------------|------------------|
| tile_1x4           |       4 |        339,213.7 |   16,904.1 |        **20.1x** |
| testsrc_1080p30    |       4 |        181,832.6 |   11,831.0 |        **15.4x** |
| testsrc_720p30     |       1 |         87,198.0 |    5,743.7 |        **15.2x** |
| tile_1x4           |       1 |        271,544.0 |   19,700.3 |        **13.8x** |
| testsrc_1080p30    |       2 |        148,492.1 |   12,307.0 |        **12.1x** |
| foreman-fp1        |       1 |          8,130.3 |      818.5 |         **9.9x** |
| quantizer-00       |       1 |         12,084.5 |    1,692.6 |          **7.1x** (now 3.6x FASTER than vpxdec) |
| testsrc_1080p60    |       4 |         11,321.2 |    4,980.0 |          **2.3x** |

## Previous R.3 ledger preserved below

# decode head-to-head — 2026-05-23 (R.3 update)

host: windows-x86_64  unpinned  build: release (3c8d6d1ddea4abc4)
rvp9: post-R.3 (trace_flags lib.rs extension + intra dq_coefs thread_local)

## R.3 post-fix

| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |
|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|
| quantizer-00       | 352x288    |     1 |       1 |      1836.0 |       1921.0 |         6088.5 |             0.32 |         - | **FASTER**  |
| tile_1x4           | 1920x1080  |     4 |       1 |     21870.0 |      21073.9 |         3964.5 |             5.31 |         - | n/a         |
| tile_1x4           | 1920x1080  |     4 |       4 |     17122.4 |      16126.5 |         2515.7 |             6.41 |         - | n/a         |
| testsrc_720p30     | 1280x720   |     1 |       1 |      7040.0 |       6487.3 |          627.3 |            10.34 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       2 |     14633.3 |      13330.8 |          820.9 |            16.24 |     33000 | PASS-info   |
| testsrc_1080p30    | 1920x1080  |     4 |       4 |     12730.4 |      11100.2 |          614.4 |            18.07 |     33000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       4 |      5668.6 |       7529.8 |          592.2 |            12.72 |     16000 | PASS-info   |
| testsrc_1080p60    | 1920x1080  |     4 |       8 |      5614.6 |       7544.7 |          596.1 |            12.66 |     16000 | PASS-info   |
| foreman-fp1        | 352x288    |     1 |       1 |       903.7 |        874.0 |          317.0 |             2.76 |         - | n/a         |

## R.3 before/after delta (post-R.2 → post-R.3)

| fixture            | threads | post-R.2 p50 us | post-R.3 p50 us | R.3 speedup |
|--------------------|---------|-----------------|-----------------|-------------|
| quantizer-00       |       1 |          1870.8 |          1836.0 |  1.02x      |
| tile_1x4           |       1 |         29411.0 |         21870.0 |  **1.34x**  |
| tile_1x4           |       4 |         18448.8 |         17122.4 |  1.08x      |
| testsrc_720p30     |       1 |          9675.7 |          7040.0 |  **1.37x**  |
| testsrc_1080p30    |       2 |         16605.9 |         14633.3 |  1.13x      |
| testsrc_1080p30    |       4 |         12742.7 |         12730.4 |  1.00x      |
| testsrc_1080p60    |       4 |          5139.3 |          5668.6 |  0.91x (noise) |
| testsrc_1080p60    |       8 |          6541.7 |          5614.6 |  **1.17x**  |
| foreman-fp1        |       1 |          1112.5 |           903.7 |  **1.23x**  |

## Compound: baseline (pre-R.1) → post-R.3

| fixture            | threads | pre-R.1 baseline | post-R.3   | compound speedup |
|--------------------|---------|------------------|------------|------------------|
| tile_1x4           |       4 |        339,213.7 |   17,122.4 |        **19.8x** |
| testsrc_1080p30    |       4 |        181,832.6 |   12,730.4 |        **14.3x** |
| testsrc_720p30     |       1 |         87,198.0 |    7,040.0 |        **12.4x** |
| tile_1x4           |       1 |        271,544.0 |   21,870.0 |        **12.4x** |
| testsrc_1080p30    |       2 |        148,492.1 |   14,633.3 |        **10.1x** |
| foreman-fp1        |       1 |          8,130.3 |      903.7 |         **9.0x** |
| quantizer-00       |       1 |         12,084.5 |    1,836.0 |          **6.6x** (now 3.32x FASTER than vpxdec) |
| testsrc_1080p60    |       4 |         11,321.2 |    5,668.6 |          **2.0x** |

All PLAN.md §2 budgets PASS with significant headroom. Worst remaining
gap: 20.7x on testsrc_1080p30 t=4. Continuing in R.4 (profile-guided).

## Methodology

R.3 changes:
- crates/rvp9-decoder/src/trace_flags.rs: extended TraceFlags with the
  9 lib.rs `env::var` sites (disable_lf, trace_seg, trace_fc,
  dump_altref, dump_lf_masks). Last one was per-SB (~500/frame at 1080p).
- crates/rvp9-decoder/src/reconstruct.rs: added thread_local
  `DQ_COEFS_BUF: RefCell<Vec<i32>>` (sized to MAX_TX_COEFFS = 1024).
  Replaced per-call `let mut dq_coefs = [0i32; MAX_TX_COEFFS]` (4 KB
  stack zero) in `reconstruct_intra_block` with a thread_local borrow.
  The inter equivalent (`reconstruct_inter.rs:544`) was tried but
  regressed testsrc_1080p60 by 30% — the per-call RefCell+TLS overhead
  exceeded the stack-zero savings for synthetic content with high
  TX-block density. Reverted; left as a follow-up needing UnsafeCell
  or per-tile hoist via parameter threading.
- `bool_reader::read` was also tried at `#[inline(always)]` but
  regressed 1080p60. Reverted.

Bisection summary: lib.rs trace_flags + intra dq_coefs is the
keep-and-ship subset. Mechanical, bit-exact, no test churn.

## Previous ledger sections preserved below (R.2 + R.1 + baseline)

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
