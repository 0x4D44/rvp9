# Decoder kernel audit — rvp9 vs libvpx (head-to-head, pinned)

**Started:** 2026-05-26
**Why:** the "Match vpxdec perf" drive measured rvp9-internal kernels
(scalar vs AVX2) and rvp9 end-to-end vs vpxdec. It never measured
rvp9 kernels vs libvpx kernels apples-to-apples. The single-thread
torture-test gap (~6× at 720p) scales with pixel count and almost
disappears on low-coefficient content (quantizer-00 t=1: rvp9 *beats*
vpxdec 4×). That fingerprint says the gap lives in the per-pixel DSP
kernels (IDCT + sub-pel MC + LF inner), not in framework overhead.

**Hypothesis (falsifiable):**
> rvp9's per-pixel DSP kernels run substantially slower than libvpx's
> on identical inputs, pinned, AVX2 vs AVX2.

If true, single-thread parity is reclaimable serial work. If false,
the gap is dispatch / bounds-checking / glue and we re-direct.

## Scope of first run

Smallest experiment that disconfirms or supports the hypothesis:

| Kernel               | rvp9 entry point                       | libvpx symbol             |
|----------------------|----------------------------------------|---------------------------|
| idct16x16, eob<=38   | `Dsp::idct16x16_38_add()`              | `vpx_idct16x16_38_add`    |
| idct32x32, eob<=135  | `Dsp::idct32x32_135_add()`             | `vpx_idct32x32_135_add`   |
| convolve8 64x64 ½-pel| `Dsp::convolve8()` (AVX2 dispatch)     | `vpx_convolve8`           |

**Apples-to-apples requirements:**
- Same i32 coefficient input on both sides (libvpx is built with
  `--enable-vp9-highbitdepth`, so `tran_low_t == int32_t`).
- Same filter table for convolve (`EIGHTTAP_KERNEL[8]` matches libvpx
  `vp9_sub_pel_filters_8[8]` by VP9 spec).
- Identical LCG-generated source bytes / coefficient values, same seed.
- Bit-exactness gate: FNV-1a 64-bit hash of dest buffer after one call;
  rvp9 and libvpx hashes MUST match before we trust the ratio.
- Both rigs pinned to CPU 0 via `bench-pin.ps1`.
- Both use min-of-N-batches after warmup (same methodology as the
  existing `dsp-microbench`).

## Shape decision (architecture log)

Considered three FFI shapes:

1. **Static link libvpx.a into Rust MSVC** — clean but `libvpx.a` is
   mingw-built. Cross-toolchain link (mingw .a → MSVC link.exe) needs
   stub helpers (`__chkstk_ms`, `___udivdi3`, libgcc unwind). Likely
   wastes a day before producing a number.
2. **Rebuild libvpx as DLL, `LoadLibrary` from Rust** — clean but
   requires reconfiguring libvpx with `--enable-shared` and the
   first-signal investment is still write-FFI-shims-for-3-kernels.
3. **Out-of-process: small mingw C harness times libvpx, rvp9 side
   times itself** — two processes, two timing rigs, but each uses
   its own toolchain natively. Same pin method, same clock (QPC vs
   `Instant`, both `monotonic_raw`-equivalent on Windows), same
   input-by-seed. Lowest setup cost.

**Picked (3) for first signal.** If the signal is unambiguous we
have the answer; if it's ambiguous (e.g. <1.5× ratios with high
variance) we invest in (2) for an in-process apples-to-apples rig.

## Checklist

- [x] Recon: rvp9 kernel signatures, libvpx dispatch entry points,
      `EIGHTTAP_KERNEL` parity, libvpx include topology, mingw
      toolchain available.
- [x] C harness `tools/libvpx_kernel_bench/main.c` — times the 3
      kernels, prints CSV `side,kernel,size,ns,input_fnv64,output_fnv64`.
- [x] `tools/libvpx_kernel_bench/Makefile` — mingw build (TMPDIR=/tmp,
      CC=gcc; links `build/libvpx/libvpx.a`).
- [x] `dsp-microbench` `vs_libvpx` case — matching rvp9 timings, same
      LCG seed, same CSV, calls through the production `Dsp` dispatch.
- [x] `scripts/run-kernel-audit.ps1` — driver: pin, run both N×, min,
      join, assert hashes equal, print ratio table, write CSV.
- [x] First pinned run captured to
      `bench/results/kernel-audit-pinned-2026-05-26.csv` (+ driver CSV
      `kernel-audit-2026-05-26T18-48-11Z.csv`).
- [x] Findings written below.

## Findings — first run (2026-05-26, pinned CPU0, min-of-5)

| kernel | size | rvp9 ns | libvpx ns | ratio | bit-exact | ISA |
|---|---|---|---|---|---|---|
| idct16x16_38_add  | 16x16 |   74.2 |   86.2 | **0.86×** | yes | rvp9 AVX2 finisher / libvpx SSE2 |
| idct32x32_135_add | 32x32 |  368.6 |  219.6 | **1.68×** | yes | rvp9 **SSE2** / libvpx **AVX2** |
| convolve8 ½-pel   | 64x64 | 1134.8 |  479.4 | **2.37×** | yes | **both AVX2** |

All three are **byte-exact** (FNV-1a 64 of input AND output match across
rigs) — the apples-to-apples guarantee holds.

### Verdict on the hypothesis

**SUPPORTED but refined.** The per-pixel kernel gap is real (1.68×,
2.37×) but **kernel-specific, not uniform** — and rvp9 *beats* libvpx
on idct16x16_38 (0.86×), which proves:

1. The framework / dispatch / call overhead is NOT the bottleneck — when
   a kernel has been invested in (AVX2 finisher), rvp9 wins.
2. The serial single-thread gap at the decode level lives in specific
   under-optimised kernels, two of which are now named with sized,
   high-confidence wins.

### Two distinct gap classes surfaced

- **Missing AVX2 path** (idct32x32_135 — rvp9 dispatches SSE2 where
  libvpx ships AVX2). 1.68× → likely closes to ~parity or a win by
  porting `vpx_idct32x32_135_add_avx2`. Clean, bounded, conformance-
  guarded work.
- **AVX2 present but slower** (convolve8 2.37×, both AVX2). The deepest
  and most valuable gap: the sub-pel MC kernel is the single largest
  per-pixel cost in inter frames and rvp9's AVX2 is 2.4× off libvpx's
  hand-tuned AVX2. This is where "beat vpxdec" is won or lost.

### Caveats on this run

- Single bench host (Arrow Lake hybrid), pinned CPU0, AboveNormal prio.
  Min-of-5; run-to-run spread < 5% on libvpx, < 8% on rvp9.
- The convolve case is 2-D 64×64 half-pel (worst-case blur, both taps
  active). Other block sizes / sub-pel positions not yet swept — the
  2.37× is the worst-case MC, not a frame-weighted average.
- idct inputs are full-density within the eob bucket (8 rows for _38,
  16 rows for _135). Real bitstream eob distribution is lighter; the
  per-call ns is an upper bound for the bucket.

## Findings — full map (2026-05-27, pinned CPU0, min-of-5)

17 kernels, **all byte-exact** across rigs. CSV:
`bench/results/kernel-audit-fullmap-2026-05-27.csv`.

### IDCT family

| kernel | rvp9 ns | libvpx ns | ratio | note |
|---|---|---|---|---|
| idct4x4_16    |  10.3 |   6.2 | 1.67× | tiny — call overhead dominates |
| idct8x8_12    |  16.2 |  13.5 | 1.20× | |
| idct8x8_64    |  21.9 |  27.7 | **0.79×** | rvp9 faster |
| idct16x16_10  |  47.8 |  55.2 | **0.87×** | rvp9 faster |
| idct16x16_38  |  76.6 |  88.3 | **0.87×** | rvp9 faster |
| idct16x16_256 | 126.9 |  77.6 | 1.64× | dense path slow vs its own buckets |
| idct32x32_34  | 322.1 | 216.0 | 1.49× | rvp9 SSE2 / libvpx AVX2 |
| idct32x32_135 | 384.3 | 219.3 | 1.75× | rvp9 SSE2 / libvpx AVX2 |
| idct32x32_1024| 577.8 | 310.2 | 1.86× | rvp9 SSE2 / libvpx AVX2 |

### Convolve8 (sub-pel MC) family

| kernel | rvp9 ns | libvpx ns | ratio | note |
|---|---|---|---|---|
| convolve8_vert 16x16  |  30.4 |  16.4 | 1.85× | |
| convolve8_vert 64x64  | 233.1 | 199.2 | **1.17×** | near parity |
| convolve8_horiz 16x16 | 168.1 |  17.9 | **9.39×** | ← pathological |
| convolve8_horiz 64x64 | 924.5 | 242.5 | 3.81× | |
| convolve8_2d 8x8      | 174.5 |  19.6 | **8.92×** | inherits horiz cost |
| convolve8_2d 16x16    | 289.7 |  41.1 | 7.05× | |
| convolve8_2d 32x32    | 501.4 | 133.3 | 3.76× | |
| convolve8_2d 64x64    | 1273.6| 481.4 | 2.65× | |

### Headline conclusions

1. **`convolve8_horiz_avx2` is the #1 target, by a wide margin.** 9.4× at
   16×16. The vertical pass is near-parity (1.17–1.85×), so the
   asymmetry is intrinsic to rvp9's *horizontal* AVX2 path — it is almost
   certainly not lane-parallel / has a hot scalar tail. Sub-pel MC is the
   dominant inter-frame per-pixel cost, and the 2-D path inherits the
   horiz cost (2D ≈ horiz + vert), so fixing horiz fixes the whole MC
   stack. Single biggest lever found for the single-thread gap.
2. **32x32 IDCT family (1.49–1.86×)** — rvp9 dispatches SSE2 where libvpx
   ships AVX2. Clean, bounded port of `vpx_idct32x32_*_add_avx2`.
3. **Dense `idct16x16_256` (1.64×)** is an oddity — slower than its own
   eob-bucket variants (_10/_38 both *beat* libvpx). Worth a look; the
   dense 16x16 path isn't sharing the buckets' fast finisher.
4. **rvp9 already wins or ties on 8x8 and 16x16 eob-bucket IDCT** —
   reconfirms the framework is sound; the gaps are specific kernels.

### Priority order for the optimize phase (frame-weighted)

1. `convolve8_horiz_avx2` rewrite — biggest single win, touches all
   sub-pel inter MC (horiz + 2D).
2. `idct32x32_*` AVX2 port — bounded, 1.5–1.9×, conformance-gated.
3. `idct16x16_256` dense AVX2 — smaller, but a clean 1.64×.
4. `convolve8_vert 16x16` (1.85×) and small-IDCT call overhead — minor.

### Reproduce

```
make -C tools/libvpx_kernel_bench            # mingw64 shell, TMPDIR=/tmp
cargo build --release -p rvp9-bench --bin dsp-microbench
pwsh -File scripts/run-kernel-audit.ps1 -Runs 5
```

## Notes for follow-up

- If signal positive: expand to LF inner kernels
  (`vpx_lpf_horizontal_8`, `vpx_lpf_vertical_8`) and the smaller IDCT
  variants (`_10`, `_256`, `_1024`). Then dispatch overhead per call:
  measure `Dsp::idct…()` getter vs direct call to localize dispatch
  cost.
- If signal weak/absent: pivot to glue-overhead profiling. The 6×
  serial gap must live somewhere — likely in the per-block dispatch,
  bounds checks in coefficient unpacking, or entropy state churn.
- Independent of result: the bidirectional bench infrastructure has
  zero overlap with the existing scalar-vs-AVX2 microbench. Keep
  both — different oracles for different questions.
