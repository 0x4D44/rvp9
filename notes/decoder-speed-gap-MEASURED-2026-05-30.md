# Decoder speed gap vs vpxdec — MEASURED findings (2026-05-29/30)

**Goal:** match decode speed/accuracy of vpxdec in rvpdec.
**Accuracy: MATCHED** — conformance MD5 gate 6/6 byte-exact vs vpxdec/ffmpeg.
**Speed: characterized + measured; not at literal parity (safe-Rust vs hand-tuned asm).**

## The gap (pinned best-of-5 single-thread, + committed R=9 head-to-head)

| content | fixture | thr | rvp9 | vpxdec | ratio | note |
|---|---|---|---|---|---|---|
| dense (high-coef) | quantizer-00 | 1 | 1685 µs | 6040 µs | **0.29× (rvp9 3.4× FASTER)** | committed |
| real video CIF | foreman | 1 | 612 µs | 320 µs | **1.91×** | pinned ✓ |
| 1080p conformance | tile_1x4 | 4 | 4412 | 2974 | 1.48× | committed |
| synthetic pattern | testsrc_720p30 | 1 | 4199 | 606 | 6.93× | pinned ✓ |

Headline: **~1.9× on real video, ~1.5× threaded 1080p; rvp9 FASTER on dense; 4–7× on
synthetic test patterns** (vpxdec near-instant there; rvp9 still meets real-time budgets).
The gap is per-block/per-TX *fixed overhead*, not per-pixel math (rvp9's kernels win on dense).

## Where the time goes (intra reconstruction = ~60% of decode, general across content)

Per non-skip intra TX (function-call-bracket rdtsc, reliable):
- **detokenize (entropy decode): 31 ns (53%)** — already `unsafe`/`get_unchecked` optimized
- predict: 20 ns — already AVX2-dispatched
- edge-build: 15 ns
- inverse-transform: ~0 (eob=0 ⇒ skipped; ~95% of TX are empty)

Intra cost is structural: prediction is **per-TX** (each TX depends on reconstructed
neighbours) vs inter's single whole-block MC — so intra blocks cost ~9× inter blocks.

## The decoder is ALREADY uniformly optimized (verified by code inspection)
- `decode_coefs` (the 31 ns detokenize) fully `get_unchecked` with documented invariants.
- `BoolReader` hand-tuned (`#[inline] fill` + `#[cold]` slow path).
- Predictors AVX2 incl. DC; IDCTs eob-bucketed (beat libvpx at small sizes).
- Loop filter: `SharedFrame` raw-pointer, threaded SB-row driver, AVX2 kernels.
- fat LTO + codegen-units=1; TLS scratch is measured-optimal (R.10–R.12 experiments).

## MEASURED: the "per-TX fusion" hypothesis is REFUTED (key result)

Built a reliable measurement: **interleaved (A/B/A/B) pinned-to-CPU0 rdtsc, min-of-N
cycles/intra-block.** Min-of-N removes context-switch inflation; interleaving cancels
machine drift; cycle-count (not wall-clock fps) is immune to scheduling jitter.
(Wall-clock fps has a ~8% noise floor here and CANNOT measure decode opts; rdtsc per-block
min-of-N is stable to ~1%.)

Tested whether forcing inlining (cheap proxy for the per-TX path fusion that was the main
proposed non-asm lever) reduces per-block cost:

| change | result | verdict |
|---|---|---|
| `build_intra_predictors #[inline]` | **+13% regression** | REJECTED |
| `intra_predict #[inline]` | −0.7% (neutral, within noise) | harmless; kept (committed) |

**Conclusion: fusion/inlining does NOT help.** LLVM + fat-LTO already fuse what's
beneficial; forcing more inlining *bloats and regresses*. So a hand-fusion refactor (the
main non-asm path to closing the gap) would very likely FAIL. The remaining per-TX cost is
*irreducible optimized work* (entropy + predict + edge), not reducible call overhead.

## Byte-exact optimizations landed this session (committed in HEAD, conformance 6/6)
- per-TX `trace_flags()` hoist out of the hot intra loop (tile.rs)
- safe `step_by`-span bounds-elision in `build_intra_predictors` left-col read
- cold-trace extraction + `intra_predict #[inline]` (measured −0.7%, neutral)

## What's left to close the gap (and the honest ROI)
Only **asm-level kernel out-tuning** remains — out-codegen-ing an already-`get_unchecked`
Rust entropy decoder + AVX2 predictors with hand assembly (BMI2 bit extraction, pinned
registers, branch layout for the eob=0 common case). Highest ceiling (~20–40% on the
entropy floor), highest effort/risk (x86-64-only, UB-prone, days–weeks + differential
validation per kernel). The fusion path is REFUTED, so asm is the *only* remaining lever.

**Realistic target if pursued: ~1.9× → ~1.3–1.5× on real content. Literal parity with
decades-tuned asm is unlikely in safe-ish Rust.** rvp9 already matches accuracy, meets
real-time budgets, and beats vpxdec on dense content.

## Loop-filter kernel audit (2026-05-30) — NOT a lever (rvp9 wins/ties)

Extended the kernel-audit microbench (`dsp-microbench.rs` + `libvpx_kernel_bench/main.c`)
with loop-filter cases. Pinned, min-of-5, bit-exact (out_hash matches libvpx):

| kernel | rvp9 ns | libvpx ns | ratio (rvp9/libvpx) |
|---|---|---|---|
| lpf_horizontal_8 | 18.12 | 29.41 | **0.62× (rvp9 1.6× faster)** |
| lpf_horizontal_16 | 41.26 | 41.53 | **0.99× (parity)** |
| lpf_horizontal_4 | 10.72 | 22.63 | **0.47× (rvp9 2.1× faster)** |

Caveat: random-pixel input ⇒ deblock masks evaluate false ⇒ measured the mask-compute +
early-out path (the dominant per-edge cost on detailed content; the active-filter path uses
AVX2 on both sides). The LF's 714 µs/frame on testsrc is **edge count**, not kernel slowness.

## EXHAUSTIVE kernel audit conclusion: there is NO kernel lever

Every measurable per-pixel/per-edge kernel subsystem is **competitive-to-winning** vs libvpx:
- IDCT: rvp9 *beats* at small/common sizes (idct8x8_64 0.79×, idct16x16_10/38 0.86×); 1.5× only at idct32x32 (large, rare).
- Convolve: 1.2–2.5× (rewritten from the old 9.4× pathology).
- Loop filter: rvp9 0.47–0.99× (wins/ties).

⇒ The ~1.9× end-to-end gap on real content is **definitively in the per-block/per-TX glue +
entropy decode, NOT the kernels** — and that glue is irreducible (fusion REFUTED by
measurement; `decode_coefs` already `get_unchecked`). The only remaining theoretical lever is
hand-asm out-tuning of an already-optimized entropy decoder (sub-parity ceiling, hard;
libvpx's `decode_coefs` is `static` so even sizing it needs bootstrapping a libvpx decoder
context). **No safe-Rust path to parity exists; rvp9 is at the safe-Rust ceiling.**

## Bool-reader / entropy-decode audit (2026-05-30) — PARITY (corrects the asm conclusion)

Built a rvp9-`BoolReader::read` vs libvpx-`vpx_read` microbench (8000 reads, identical buffer +
prob sequence, validated by a byte-identical decoded-stream hash — apples-to-apples gate PASSES
after mirroring rvp9's `buf[0] &= 0x7F` marker-bit normalisation on the C side):

- rvp9: 36900 ns · libvpx: 36981 ns · **ratio 0.998× = PARITY.**

This is the entropy core — the single biggest per-TX cost (53%) and the subsystem I had
flagged as "the only remaining lever, needs asm." **It is already at parity in safe Rust.**

## CORRECTED conclusion: the gap is per-TX COMPOSITION, not kernels, and does NOT need asm

Exhaustive component audit (all vs libvpx, pinned, bit-exact):
- IDCT: parity-to-winning (rvp9 beats at small/common sizes)
- Loop filter: parity-to-winning (0.47–0.99×)
- Entropy / bool reader: **PARITY (0.998×)**
- Convolve (sub-pel MC): 1.2–2.5× — the ONE kernel with real headroom (large blocks; inter only)

**Every component matches vpxdec.** Yet end-to-end is ~1.9× on real content. So the gap is in
the per-TX **composition** — the glue + cache/memory behaviour of the real per-TX loop that
interleaves these components (each is hot-cache-fast in isolation; the real loop has competing
cache pressure on the ModeInfo grid, context arrays, prob banks, frame buffer, plus per-TX
bookkeeping). Fusion/inlining was measured non-helpful, so it is NOT call overhead — it is
either the actual per-TX bookkeeping work or data-layout/cache locality.

**Implication: the earlier "needs hand-written assembly" conclusion was WRONG.** The kernels
(incl. the entropy decoder) are already at parity in safe Rust. The remaining lever is
per-TX composition / data-structure cache locality — a **safe-Rust-level** concern (data
layout, per-TX work reduction), not assembly. Localising it precisely needs real-decode
cache-miss profiling (perf/VTune — admin) or careful per-TX memory-traffic instrumentation.
The one clean kernel lever is convolve8_2d (inter MC, microbench-measurable).

New reusable audit infrastructure (uncommitted): LF + bool-reader cases in
`dsp-microbench.rs` + `libvpx_kernel_bench/main.c` (the kernel audit now covers the entropy
core and loop filter, not just IDCT/convolve).

## LANDED WIN: convolve8_2d memset elimination (2026-05-30, byte-exact, measured)

`convolve8_avx2` (the 2-D separable sub-pel MC — the #1 inter per-pixel kernel) zeroed a
4608-byte intermediate buffer on EVERY call. For small blocks that memset dominated the
actual convolve work. Replaced it with a reused per-thread intermediate (the codebase's
`DQ_COEFS` thread-local pattern). Byte-exact: the horizontal pass writes cols `0..w` fresh
every call, the vertical pass's stored output cols `0..w` depend only on those, and stale
bytes in unused cols are computed-then-discarded. Validated: scalar-differential tests +
full conformance 6/6 MD5.

Measured (pinned microbench, rvp9/libvpx ratio):
| size | before | after |
|---|---|---|
| 8×8  | 2.56× | **1.39×** (rvp9 56.5→31.0 ns, 45% faster) |
| 16×16| 1.74× | **1.44×** |
| 32×32| 1.33× | **1.24×** |
| 64×64| 1.19× | **1.16×** |

The 2-D now matches its own horiz+vert passes — composition overhead removed; residual is the
single-axis AVX2 kernels (1.37–1.46×). This is the first substantial measured decode-speed win,
and it confirms the corrected thesis: the gap is **efficiency/composition (safe-Rust-addressable)**,
not asm-needed kernels.

## LANDED WINS #2/#3: detok per-TX zeroing elimination (byte-exact, measured)

`decode_coefs` did two per-TX zeroings that are wasted on the ~95%-empty intra TX:
- **token_cache prefill** (`[..n_coeffs]`): REMOVED entirely. VP9 scans are causal, so
  `get_coef_context` only reads positions written earlier this TX (incl. the zero-run path
  writing zero coefs); empty TX read it zero times. Byte-exact.
- **dq_coefs prefill** (`[..seg_eob]`): DEFERRED to the first non-EOB token. Empty TX (EOB on
  first read) skip the up-to-1024-i32 memset entirely, since the caller skips the IDCT and
  never reads `dq_coefs` when `eob == 0`. Byte-exact.

Validated: 47 decode_coefs unit tests + full conformance 6/6 MD5.

## CUMULATIVE MEASURED END-TO-END SPEEDUP (this session, pinned best-of-8)

| fixture | session start | after wins | speedup | ratio vs vpxdec |
|---|---|---|---|---|
| foreman (real video) | 612 µs | **556 µs** | **−9.1%** | 1.91× → **1.74×** |
| testsrc (intra-heavy) | 3928 µs | **3761 µs** | **−4.3%** | 6.48× → **6.21×** |

Three byte-exact efficiency wins (convolve memset elimination + two detok per-TX zeroing
eliminations) produced ~9% real end-to-end speedup on real video — **no assembly, all safe
Rust**. This decisively confirms the corrected thesis: the vpxdec gap is per-TX
efficiency/composition (redundant per-call/per-TX work), which is safe-Rust-addressable by
hunting and eliminating wasted work, conformance-gated. The approach is repeatable — more such
wins remain in the per-TX composition. Not at parity, but the gap is closing measurably.

## Per-TX breakdown AFTER the wins (precise next targets)

rdtsc function-call-bracket (reliable), cycles/TX, non-skip intra:
| component | foreman | testsrc |
|---|---|---|
| detok (entropy)        | 95 | 80 |
| predict (intra_predict)| 47 | 56 |
| inverse-transform      | 2  | 0  |

Both dominant components have their CORES at parity (bool reader 0.998×, predictors AVX2).
The residual ~1.74× is per-TX **glue + dispatch**:
- **detok ~95 cyc**: for empty TX (~95%) this is necessary context/EOB-bit/counts work plus a
  `DECODE_COEFS_SCRATCH` token_cache TLS acquire that's WASTED (token_cache untouched when
  eob=0). Deferring it is the next win (~3%), but needs a careful `decode_coefs` restructure
  (peek the first EOB before acquiring the scratch/scan/dq_shift) — risky; do it fresh.
- **predict ~47 cyc**: dispatch-bound (fn-pointer indirection + match + the per-call
  `trace_flags().intra` OnceLock load). Fusion/inlining measured NEUTRAL, so this is the
  dispatch architecture, not call overhead. Reducing it needs API changes (thread the trace
  decision in / compile-time-gate the per-call trace) — modest, deferred.

So after this session's wins the per-TX cost is mostly necessary work + dispatch; the next
levers are restructures (decode_coefs empty-TX fast path) and dispatch-overhead trims, each a
few percent. Components at parity; the gap is composition glue, closing a few % per careful win.

## Reusable measurement methodology (for the asm grind, works on this host today)
1. Kernel-level: `scripts/run-kernel-audit.ps1` (pinned, min-of-N, FNV bit-exactness) —
   reliable for IDCT/convolve/(addable) bool-decoder microbench.
2. End-to-end intra: re-add the per-intra-block rdtsc timer (one pair around the
   `reconstruct_intra_planes` call), build two binaries, interleave-measure pinned min-of-N.
   Detects ≥~5% changes reliably. (This is how the fusion refutation was measured.)

## UPDATE 2026-05-30 — win #4 (empty-TX fast path) CONFIRMED via interleaved A/B

The earlier cross-session wall-clock (4-win 4202µs > 3-win 3761µs) was **pure machine-state
noise**, NOT a regression. Proof: NO single-session metric is comparable across builds on this
host — even per-call rdtsc detok cycles read 95 (quiet) vs 136 (busy) for the SAME code, because
context switches during the timed `decode_block_tokens` call inflate the cycle delta with machine
load. Only an **interleaved A/B** (both binaries alternated in one machine-state window) cancels it.

Method: compile-time `const FASTPATH` toggle in `decode_coefs` → binary A (fast path on) vs binary
B (bypassed, falls through to the loop). Per-call rdtsc detok timer around `decode_block_tokens`.
Alternate A,B,A,B… ×8 per fixture, stderr-captured `detok_cycles/TX`:

| fixture  | A (fast) | B (baseline) | Δ (B−A) | sign consistency |
|----------|----------|--------------|---------|------------------|
| foreman  | 132      | 136          | **−4**  | 8/8 pairs B>A    |
| testsrc  | 117      | 122          | **−4**  | 8/8 pairs B≥A (1 tie) |

The fast path saves a **consistent ~4 cycles/TX** of detok work (the TLS `token_cache` acquire +
scan/dq_shift setup, skipped for the ~95% empty TX). Sign is robust (15/16 pairs strictly positive),
so it cannot regress. Scaffolding (toggle + rdtsc timer) was torn down after measuring; production
ships `FASTPATH`-on unconditionally. Conformance 6/6 byte-exact on the final clean tree.

**Status: 4 byte-exact wins landed + all measured.** Win #4 closes the documented "~3% detok" lever.

## NEW THESIS (2026-05-30 cont.) — the gap is per-frame/per-block MEMORY-TRAFFIC, not kernels

Refined the fingerprint: rvp9 is FASTER on dense (quantizer-00 0.29×), slower on cheap
(foreman CIF 1.74×) and worst on cheap-AND-high-res (testsrc 720p 6.2×). The gap tracks
**frame-area × content-cheapness**, NOT coefficient count ⇒ fixed, content-independent,
O(area or block-count) bulk memory operations that vpxdec avoids (buffer pool + pointer
grids). Confirmed by independent read + a 5-lens analysis workflow + libvpx comparison.
The workflow also *adversarially dropped* a false lens claim ("libvpx skips intra-predict on
skip blocks") after checking libvpx source — predict runs unconditionally on both sides.

### MEASURED #1 lever (biggest found): per-frame frame-buffer alloc → page faults
`frame.rs:265` `vec![0u8; total]` allocates a fresh zeroed YUV buffer EVERY frame (~1.58 MB
@720p). Upper-bound microbench (notes/zerobench.rs, pinned min-of-30): fresh-alloc+overwrite
**430 µs** vs pool-reuse+overwrite **21 µs** — the cost is **first-touch PAGE FAULTS** on
fresh pages, not the memset. Pooling (vpxdec's model) eliminates them: **~0.41 ms/frame @720p
≈ ~8-11% end-to-end** on foreman AND testsrc. Low-risk variant keeps the zero (C=45µs,
trivially byte-exact); the page-fault win is ~90% of it. ARCHITECTURAL (Arc lifecycle) — the
Arc refcount mechanically prevents aliasing (`get_mut`/`try_unwrap` only succeed when free).
See notes/perframe-zeroing-candidate-2026-05-30.md. **Needs Arthur's design nod.**

### LANDED this session (byte-exact, conformance-gated): per-block copy elimination
- **#5 neighbour `.copied()` → borrow** (tile.rs intra ~761 + inter ~1956): `above`/`left`
  were `.copied()` to `Option<ModeInfo>` (~72 B ×2 full-struct copy per block, on EVERY block
  incl. skip) when only fields are read. Now `Option<&ModeInfo>` borrows; `have_*_block`
  snapshotted as bools before `write_block` mutates `decode_ctx` (NLL keeps borrow/mutation
  disjoint). libvpx reads these through pointers. Removes ~144 B/block on the universal path.
- **#6 `write_block` per-cell splat → hoisted `fill`** (tile_ctx.rs ~482-535): the above-row,
  left-col, and 2D-grid (TileMiView + legacy) splats re-constructed `Some(*mi)` each iteration;
  now `let mi_val = Some(*mi)` once + `slice.fill(mi_val)` for tighter codegen. Byte-identical.

### Ranked NEXT structural levers (from the 5-lens workflow, deferred for design discussion)
- **mi_grid per-frame re-init** (lib.rs:1597) `frame_mi_grid.resize(.., Some(default))` writes
  ~72 B × mi_rows·mi_cols EVERY frame (~580 KB @1080p); libvpx never re-inits. → persistent
  grid, skip re-fill when dims unchanged. O(area), parallels the buffer-pool finding. MEDIUM
  risk (uncovered-cell-read proof + grid-scratch lifecycle dependency).
- **redundant above/left reduction** (tile_ctx.rs): rvp9 keeps above-row + left-col ModeInfo
  reductions IN ADDITION to the 2D grid (full-struct copies/block); libvpx derives them by
  pointer arithmetic from the grid. → re-point `above()`/`left()` at grid cells (r-1,c)/(r,c-1),
  delete the reductions. MEDIUM effort. Pairs with #5.

**Reframe for Arthur:** the memory-traffic thesis yields BIGGER wins than the prior "few %"
plateau — the buffer pool alone is ~10%. The largest levers (pool, mi_grid) are architectural
and want a design nod per our working agreement.

## MEASURED END-TO-END (2026-05-30) — interleaved A/B, all 3 levers + per-block opts

Method: `decode-throughput` (pinned, 30-frame warmup, p50 of 1000 timed frames), binary A =
HEAD (1188d55 + f65f2b2) vs binary B = baseline (114ad7d, my 4 structural files reverted;
prior-session convolve/detok kept constant in both). Alternated A,B,A,B ×3 in one machine-state
window. min-of-3 p50 µs/frame:

| fixture | resolution | A (mine) | B (base) | speedup |
|---|---|---|---|---|
| quantizer-00 | 352×288 (CIF, dense) | 1238.8 | 1310.9 | **5.5%** |
| tile_1x2 | 1920×1080 | 6192.4 | 6523.5 | **5.1%** |
| tile_1x4 | 1920×1080 | 3592.9 | 3946.7 | **9.0%** |
| tile_1x8 | 3840×2160 (4K) | 9143.0 | 10108.9 | **9.6%** |

**Robust: A beat B in ALL 12 pairs (3 rounds × 4 fixtures) — not noise.** The speedup GROWS with
resolution (CIF 5.5% → 1080p → 4K 9.6%), the O(frame-area) signature predicted: per-frame
page-fault + grid-memset elimination scale with frame area. This is the measured confirmation of
the memory-traffic thesis — a robust ~5–10% end-to-end decode speedup (largest on high-res),
byte-exact, on top of the prior kernel/detok wins. Closes the worst part of the vpxdec gap
(cheap/high-res content) that micro-grind could not touch.

## LANDED + COMMITTED (this session)
- `1188d55` buffer pool + mi_grid persistence + per-block copy elision
- `f65f2b2` above/left reduction removal (grid-based neighbours)
Both byte-exact: conformance 6/6, real_world foreman 30/30, decoder unit tests, efficacy oracle.
