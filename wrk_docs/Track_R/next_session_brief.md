# Track R — Next-session brief

**Date written:** 2026-05-24
**Purpose:** Capture the concrete remaining work to close the
rvp9-decoder vs vpxdec performance gap. Single-iteration mechanical
optimization (Track R, 11 waves) reached the asymptote at R.16;
further closure requires scope-bounded structural work.

## Current state (post-R.16 commit b4dfaea)

All PLAN.md §2 real-time gates PASS with 70-83% headroom. Per-fixture
ratio vs vpxdec:

| Fixture | rvp9 p50 | vpxdec mean | ratio | Comment |
|---|---|---|---|---|
| quantizer-00 | 1.71 ms | 6.09 ms | **0.28× (FASTER)** | Small fixture; rvp9 wins |
| foreman-fp1 | 0.71 ms | 0.32 ms | 2.28× | Real content |
| tile_1x4 1080p t=4 | 9.36 ms | 2.52 ms | 3.72× | Real content, multi-tile |
| testsrc_720p30 t=1 | 5.41 ms | 0.63 ms | 8.59× | Synthetic, single-thread |
| testsrc_1080p60 t=8 | 4.73 ms | 0.60 ms | 7.94× | Synthetic, max threads |
| testsrc_1080p30 t=4 | 5.73 ms | 0.61 ms | 9.33× | Worst remaining |

## Why mechanical iteration can't close this

The remaining gap on synthetic content is dominated by the
**arithmetic decoder (bitreader) hot loop** and the **coefficient
decode pipeline**. libvpx ships hand-tuned SSE2/AVX2 assembly for
these (`vpx_dsp/x86/bitreader_buffer.c` patterns + the `vpx_idct*`
SSE2 kernels). The Rust intrinsics path goes through LLVM, which
produces reasonable but not asm-quality code.

Six of the nine attempts post-R.10 reverted because the bench
noise floor (13-21%) makes small wins indistinguishable from
regressions. Without a profiler (samply needs admin; sandbox
denies), the next changes are blind tweaks with ~50% revert rate.

## Three scope-bounded follow-up tasks

### Task A — Port `vpx_idct8x8_12_add_sse2` to Rust SSE2 intrinsics

**Estimated effort:** 4 hours focused work.
**Expected impact:** 5-15% on testsrc fixtures (Tx8x8 inter blocks
with eob<=12 are very common).

**libvpx reference:**
- `third_party/libvpx/vpx_dsp/x86/inv_txfm_sse2.c:174-185`
  (`vpx_idct8x8_12_add_sse2` outer)
- `third_party/libvpx/vpx_dsp/x86/inv_txfm_sse2.h:285-341`
  (`idct8x8_12_add_kernel_sse2` inner — 56 LOC of SSE2 i16 lanes)

**Key challenges:**
- libvpx uses `tran_low_t` which is `int16_t` by default. Our
  coefficients are `i32`. Saturating-pack i32→i16 at load is safe
  because spec bounds; verify with proptest.
- `_mm_packs_epi32` (libvpx) is saturating. Our existing AVX2 i32
  kernels avoid this because proptest exercises i16::MAX synthetic
  inputs that saturate-pack incorrectly. The new _12 SSE2 kernel
  should match the C reference for in-bounds bitstream inputs and
  is acceptable for production paths (matching vpxdec behavior).
- Transpose: libvpx uses `transpose_16bit_4x4` (4×4 of i16 lanes).
  Our existing code uses `transpose8x8_i32`.

**Validation strategy:**
1. Add `idct8x8_12_add_sse2` Rust function.
2. Compare against scalar reference `idct8x8_64_add` for inputs
   with rows 4-7 forced to zero, across N=1000 random fuzz inputs
   (existing test pattern in `inverse.rs` tests).
3. Wire into dispatch slot `idct8x8_12_add` (slot exists from R.6 attempt; was reverted, would need re-adding).
4. Wire intra + inter call sites (current TODO comments at
   `reconstruct.rs:411` and `reconstruct_inter.rs:818`).
5. Conformance gate: 201/201 official + Profile-1 triangulated.
6. Bench gate: testsrc_1080p30 t=4 must improve >=5% (above noise).

**Risk:** SIMD bugs are easy. Saturation semantics differ. Triangulated
oracle (vpxdec ∧ ffmpeg) is the load-bearing safety net.

### Task B — Port libvpx's bitreader assembly to Rust intrinsics

**Estimated effort:** 1 focused day.
**Expected impact:** 30-50% on coefficient-heavy fixtures (likely
the biggest remaining lever).

**libvpx reference:**
- `third_party/libvpx/vpx_dsp/bitreader.h` (`vpx_read`,
  `vpx_reader_fill`, the renormalization step)
- The actual asm tuning is in how libvpx structures the per-bit
  hot loop with macro fusion. There's no separate .asm file —
  the C code with `__attribute__((always_inline))` compiles to
  the tight loop.

**What to port:**
- Our existing `BoolReader::read` already has LZCNT (R.7) and
  `from_be_bytes` (R.8). Further wins need:
  - Inlining the `fill()` body INTO the `read()` call site WITHOUT
    causing icache bloat (R.15 failed naïve `#[inline]`). Need to
    extract the slow path into a `#[cold]` function and inline only
    the fast path.
  - Possibly hand-write a Rust intrinsics variant that uses BMI2
    `BLSR` / `PEXT` instructions for the renormalization, replacing
    the `leading_zeros + shift` sequence.

**Validation strategy:**
- Existing conformance corpus + proptest in `bool_reader.rs`.

**Risk:** This touches the most-called function in the decoder.
Each bench iteration takes 5 minutes; debugging is slow without
samply.

### Task C — Statistical bench framework

**Estimated effort:** Half-day.
**Expected impact:** Indirect (enables detection of small wins
that are currently lost in 13-21% noise).

**Approach:**
- Run each fixture × threads combination N=5 times.
- Compute Mann-Whitney U statistic vs baseline.
- Report `p` value; gate on p < 0.05 for "real win" claim.
- Append all runs (not just one) to the CSV.

**Why it matters:**
- R.5 (IDCT eob bucketing) showed 1.03-1.07× wins on some fixtures
  and 0.91-0.94× on others. Real or noise?
- R.7 (LZCNT) is algorithmically guaranteed faster but bench
  numbers were mixed.
- R.10 was so big it was clearly real; smaller wins need statistics.

**Risk:** None — pure additive infrastructure.

## Recommended sequence

1. **Task C first** (half-day). Without it, all subsequent
   measurements are guesses.
2. **Task A** (half-day). Concrete kernel port; bit-exact-verifiable.
3. **Task B** (1 day). Highest impact but highest risk; deserves
   the careful bench framework from Task C.

After this sequence, the synthetic content gap to vpxdec should
close to ~2-3× (matching the current real-content gap). Further
closure (parity-or-better with vpxdec) likely requires:
- Hand-written inline assembly for the bitreader fast path
  (~2-3 weeks of focused work; only worth it if perf is gate
  for shipping)

## What NOT to chase in mechanical iteration

Based on Track R's revert pattern (6 of 9 post-R.10 attempts):

- `#[inline(always)]` on hot functions (R.5, R.15) — code bloat.
- `lto = "fat"` (R.5b) — interaction with `target-cpu=native`.
- `panic = "abort"` (R.5b) — runtime metadata not the bottleneck.
- MaybeUninit on RECURSIVE callers (R.11/R.12) — stack pressure
  exceeds savings.
- `get_ref` swap on `inter_predictor` (R.13) — broke LLVM analysis.
- Scalar fast paths that REPLACE existing AVX2 dispatch (R.6) —
  scalar always loses to lane-parallel AVX2.

## Files to read first in next session

- `wrk_journals/2026.05.23 - JRN - Track R decoder perf investigation.md`
  (this session's complete record)
- `bench/results/decode-headtohead.md` (R.16 final ledger)
- `bench/results/decode-headtohead.csv` (full trend across 14
  bench runs)
- `crates/rvp9-core/src/dsp/x86_64/idct8x8_avx2.rs` (existing
  AVX2 idct8x8 — pattern to mirror)
- `third_party/libvpx/vpx_dsp/x86/inv_txfm_sse2.h:285-341`
  (libvpx idct8x8_12 SSE2 kernel — port target)

## Bit-exact gate (don't compromise)

Every commit MUST pass:
- `cargo test -p rvp9-harness --test conformance --release` (201/201)
- `cargo test -p rvp9-harness --test profile1_decode --release`
- `cargo test -p rvp9-harness --test frame_parallel_regression --release`
- `cargo test -p rvp9-harness --test tile_threading --release`

These are the load-bearing safety net for any SIMD changes.
