# 00-overview.md — Stage 1 cross-cutting synthesis

**Author:** Stage 1 driver (manual fallback while async A8 ran).
**Date:** 2026-05-16.
**Sources:** A1 (3081 ll), A2 (2311 ll), A3 (1896 ll), A4 (4556 ll
[rewritten by async agent after a truncated first pass]), A5 (1979 ll),
A6 (2002 ll), A7 (32 challenges, manual fallback).

> **Note on provenance:** This synthesis was written manually while the
> async A8 agent (agentId `ab5db17fcdd6cdf3c`) ran in background. When
> async A8 completes, it may produce an alternative 00-overview.md.
> Compare/merge whichever is better.

---

## 1. Architecture in one diagram

The libvpx VP9 real-time encoder, at cpu-used 6–8, decomposed by
Stage 1 specialist domain:

```
                    ┌─────────────────────────────────────────────┐
                    │  Public iface  (vp9_cx_iface.c)             │  ← A1
                    │  vpx_codec_init / encode / get_data / del   │
                    └────────────────┬────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────┐
                    │  Encoder lifecycle (vp9_encoder.c)          │  ← A1
                    │  vp9_create_compressor → encode → remove    │
                    └────────────────┬────────────────────────────┘
                                     │
        ┌────────────────────────────┴─────────────────────────────┐
        │                                                          │
        │   Per-frame frame loop (vp9_get_compressed_data)        │  ← A1
        │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
        │   │  Rate ctrl  │  │   SVC layer │  │ Speed features  │  │  ← A5, A6
        │   │  Q + budget │  │   selection │  │ (cpu-used 6-8)  │  │
        │   └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
        │          │                │                  │           │
        │          └────────────────┴──────────────────┘           │
        │                            │                             │
        │   ┌────────────────────────▼────────────────────────────┐│
        │   │  encode_frame_to_data_rate → encode_without_recode  ││  ← A1
        │   │  (real-time path; recode loop is disabled)          ││
        │   └────────────────────────┬────────────────────────────┘│
        │                            │                             │
        │   ┌────────────────────────▼────────────────────────────┐│
        │   │  Tile-MT dispatch (vp9_encode_tiles_mt / row_mt)    ││  ← A6
        │   │  Workers process tile columns in parallel           ││
        │   └────────────────────────┬────────────────────────────┘│
        │                            │                             │
        │   ┌────────────────────────▼────────────────────────────┐│
        │   │  Per-SB encode (encode_rt_block)                    ││  ← A2
        │   │   ┌──────────────┐  ┌──────────────┐  ┌───────────┐ ││
        │   │   │ choose_       │  │ vp9_pick_    │  │ Denoiser  │ ││  ← A2, A3, A5
        │   │   │ partitioning  │→ │ inter_mode   │→ │ + AQ      │ ││
        │   │   │ (var-based)   │  │ (non-RD)     │  │ segment   │ ││
        │   │   └──────────────┘  └──────┬───────┘  └───────────┘ ││
        │   │                            │                         ││
        │   │                            ▼                         ││
        │   │           ┌──────────────────────────────┐           ││
        │   │           │  ME (vp9_full_pixel_search)  │           ││  ← A3
        │   │           │  + sub-pel refinement        │           ││
        │   │           └──────────────┬───────────────┘           ││
        │   │                          │                           ││
        │   │                          ▼                           ││
        │   │           ┌──────────────────────────────┐           ││
        │   │           │  Transform + Quantize + Tok  │           ││  ← A4 (Q, Tok)
        │   │           │  (skip ladders, optimize_b   │           ││
        │   │           │  bypassed at cpu-used ≥ 5)   │           ││
        │   │           └──────────────┬───────────────┘           ││
        │   │                          │                           ││
        │   │                          ▼                           ││
        │   │           ┌──────────────────────────────┐           ││
        │   │           │  encode_breakout_test +      │           ││  ← A2
        │   │           │  adaptive RD threshold gate  │           ││
        │   │           └──────────────────────────────┘           ││
        │   └─────────────────────────────────────────────────────┘│
        │                            │                             │
        │   ┌────────────────────────▼────────────────────────────┐│
        │   │  Loop filter + per-frame deblocking strength        ││  ← A1
        │   └────────────────────────┬────────────────────────────┘│
        │                            │                             │
        │   ┌────────────────────────▼────────────────────────────┐│
        │   │  Bitstream pack (vp9_pack_bitstream)                ││  ← A4
        │   │  Uncompressed header → Compressed header → Tile data││
        │   │  Boolean arithmetic coder, scalar throughout         ││
        │   └────────────────────────┬────────────────────────────┘│
        │                            │                             │
        │   ┌────────────────────────▼────────────────────────────┐│
        │   │  Post-encode RC update + vp9_encodedframe_overshoot  ││  ← A5
        │   │  Frame-drop decision + reference-frame rotation     ││
        │   └─────────────────────────────────────────────────────┘│
        │                                                          │
        └──────────────────────────────────────────────────────────┘

  BufferPool (12 slots: 8 ref + 1 new + 3 scaled)  ← A1
  lives OUTSIDE VP9_COMP (allocated in cx_iface.c:1126)
```

Specialist domains:
- **A1** owns the encoder shell, `VP9_COMP` god-struct, frame loop.
- **A2** owns non-RD mode decision, partition selection, skip ladders.
- **A3** owns motion search + sub-pel refinement.
- **A4** owns bitstream layout + entropy coder + ALL probability tables.
- **A5** owns rate control + denoiser + AQ.
- **A6** owns threading model + SVC layers + speed-feature dispatch.

---

## 2. Per-frame timeline (one inter frame at cpu-used 7)

The execution skeleton the Stage 4a MVP must implement:

```
T=0   vpx_codec_encode() called                              [A1]
T+1   vp9_receive_raw_frame: push source into lookahead      [A1]
T+2   vp9_get_compressed_data: pop frame, set up encode      [A1]
T+3     pick_quickcompress_mode (RT vs GOOD)                 [A1]
T+4     vp9_change_config if dimensions/bitrate changed      [A1]
T+5     vp9_rc_get_one_pass_cbr_params: set frame target     [A5]
T+6     vp9_scene_detection_onepass: ALWAYS runs             [A5]
T+7     set_rt_speed_feature_framesize_*: speed features     [A6]
        (depends on frame_type, base_qindex, SVC, source-SAD,
         noise estimate, previous-frame deadline)
T+8     vp9_rc_pick_q_and_bounds_one_pass_cbr: pick Q        [A5]
T+9     cyclic_refresh_update_*: AQ mode 3 segment map       [A5]
T+10    encode_frame_to_data_rate → encode_without_recode    [A1]
T+11    vp9_encode_frame → encode_frame_internal             [A1]
T+12    init_tile_data: prepare tile workers                 [A6]
T+13    vp9_encode_tiles_mt: dispatch to workers             [A6]
        ──── PER WORKER, PER SB ────
T+14a   choose_partitioning: variance-based partition tree   [A2]
        (force_split[21] bitmap, three-layer decision)
T+14b   nonrd_use_partition: descend partition tree          [A2]
T+14c   vp9_pick_inter_mode: walk 12-entry ref_mode_set      [A2]
        Per mode candidate:
T+14d     ME: vp9_full_pixel_search (fast_dia/hex/diamond)   [A3]
T+14e     Sub-pel: find_fractional_mv_step                   [A3]
T+14f     model_rd_for_sb_y_large: Laplacian R-D estimate    [A2]
        Skip ladders:
T+14g     SKIP_TXFM_AC_DC flag from variance test            [A2]
T+14h     encode_breakout_test: Y+U+V variance               [A2]
T+14i     adaptive RD threshold gate                         [A2]
T+14j   Best mode chosen. block_yrd: Hadamard+SATD if needed [A2]
T+14k   Denoiser per-block decision                          [A5]
T+14l   Quantize + tokenize                                  [A4]
        ──── END PER WORKER ────
T+15    All workers complete; accumulate counts back to main [A6]
T+16    Loop filter level selection                          [A1]
T+17    vp9_pack_bitstream: serialize frame                  [A4]
T+18    vp9_encodedframe_overshoot: tentative-pack check     [A5]
        (may force *q to worst_quality and re-encode)
T+19    Frame complete. Update reference-frame map.          [A1]
T+20    vp9_rc_postencode_update: RC state mutation          [A5]
T+21    update_buffer_level, bits_off_target adjustment      [A5]
T+22    Return compressed frame to caller                    [A1]
```

Notable serial dependencies:
- T+7 (speed features) depends on T+6 (scene detection).
- T+8 (Q pick) depends on T+7 (`sf->static_segmentation`).
- T+13–T+15 (parallel tile encode) requires T+12 (init_tile_data).
- T+18 (overshoot) can REWIND state and cause re-encode at worst_quality.

---

## 3. State graph

Where each piece of state lives, by lifetime and sync class:

### Per-encoder, persistent (lives `vp9_create_compressor` → `vp9_remove_compressor`):

| State | Owner | Sync | Notes |
|---|---|---|---|
| `VP9_COMP` (the god-struct, ~120 fields) | main thread | reads from workers | A1 §2 |
| `BufferPool` | external to VP9_COMP | ref-counted | A1 §7.1 — 12 slots |
| `lookahead_ctx` | main thread | mutated only by main | A1 §2 |
| `RateControl` | main thread | written pre/post frame | A5 §2 |
| `SVC` + `LAYER_CONTEXT[]` | main thread | written across layer iterations | A6 §2 |
| `FRAME_CONTEXT` (probability tables) | main thread | read by workers | A4 §2.6 |
| `VP9_DENOISER` | main thread | written between frames | A5 §2 |
| `NOISE_ESTIMATE` | main thread | written per-frame | A5 §2 |
| Cyclic refresh map | main thread | read by workers | A5 §2 |
| `SPEED_FEATURES` | main thread | written per-frame in recode loop | A6 §2 |
| Worker pool | main thread | lazy alloc | A6 §3 + A1 §7.5 |

### Per-frame, transient (alive within one `vp9_get_compressed_data`):

| State | Owner | Sync | Notes |
|---|---|---|---|
| Source `YV12_BUFFER` (`*Source`) | main thread | read by workers | A1 |
| New encoded `YV12_BUFFER` (`new_fb`) | workers WRITE | post-frame read by main | A1 |
| Per-tile contexts (`TileDataEnc`) | main thread | one per tile | A6 §2 |
| `td->mb` (per-worker MACROBLOCK) | worker | copied OUT from `cpi->td.mb` | A6 §7.2 |
| `td->counts` (per-worker FRAME_COUNTS) | worker | accumulated BACK after frame | A6 §7.4 |
| `td->rd_counts` | worker | accumulated BACK | A6 §7.4 |
| Tile-row sync (row_mt) | row workers | barrier-style | A6 §3 |

### Per-block, ephemeral:

| State | Owner | Notes |
|---|---|---|
| `pred_buffer[4]` | per-block | A2 §7.13 — `reuse_inter_pred_sby` 4-entry pool |
| Mode candidates (12+ in ref_mode_set) | per-block | A2 §4 |
| Variance tree | per-SB | A2 §7.1 |
| MV refs / candidates | per-block | A3 §4 |

### Critical aliasing invariants (mark for Stage 2 HLD-threading):

1. **`cpi->td.mb` → `worker_data->td->mb` is COPIED** at frame start
   (`vp9_ethread.c:209-211`). Workers do not share `mb` with main.
2. **`td->counts` accumulated BACK** to `cpi->td.counts` at frame end
   (`vp9_ethread.c:244`).
3. **`FRAME_CONTEXT` (probability tables) read-only-after-init within a
   frame.** Updated by main thread between frames via backward adaptation.
4. **`BufferPool` ref-counted**; workers HOLD references during encode,
   release at frame end.
5. **`adaptive_rd_thresh_row_mt = 0` at cpu-used 6 default** makes the
   adaptive-RD threshold update racy. A6 recommends bit-exact-by-default
   (set to 1 even at cpu-used 6) — R-08 in decisions-needed.md.

---

## 4. Cross-cutting concerns

### 4.1 Speed features (A6) consumed across all other domains

`SPEED_FEATURES` is a ~100-field struct set per-frame in
`set_rt_speed_feature_framesize_*` (A6 §3). Its fields are read by:

- **A2 (mode decision):** `sf->inter_mode_mask`, `sf->use_nonrd_pick_mode`,
  `sf->partition_search_type`, `sf->disable_split_mask`, `sf->mv.subpel_iters_per_step`.
- **A3 (motion estimation):** `sf->mv.search_method`, `sf->mv.subpel_search_method`,
  `sf->mv.subpel_force_stop`, `sf->mv.fullpel_search_step_param`.
- **A5 (RC):** `sf->use_simple_block_yrd`, `sf->static_segmentation`.
- **A4 (entropy):** `sf->optimize_coefficients` (always 0 at cpu-used ≥ 5).

**Implication:** the Rust port's `SpeedFeatures` type must be readable
from every subsystem. **Recommended pattern:** immutable snapshot per
frame, set by main thread before tile dispatch, read-only from workers
(D-10 in decisions-needed.md).

### 4.2 SVC layer context (A6) consumed by RC + frame loop

A6's `LAYER_CONTEXT[]` is consumed by:

- **A1 (frame loop):** layer iteration order, layer-specific buffer/QP.
- **A5 (RC):** per-layer rate control state (`buffer_level`,
  `rate_correction_factors`, `bits_off_target` per layer).
- **A6:** reference-frame mapping per `(temporal, spatial)` layer.

**Implication:** SVC layer state must be partitionable per layer, with
RC and frame-loop both reading the same `LAYER_CONTEXT` (no duplication).

### 4.3 Frame contexts / probability updates (A4) consumed by mode decision + threading

`FRAME_CONTEXT` is read by:

- **A2 (mode decision):** for non-RD mode-cost approximation.
- **A4 (bitstream):** for actual encoding probabilities.
- **A6 (worker dispatch):** each worker gets a read-only handle to the
  frame's `FRAME_CONTEXT` plus a private `FRAME_COUNTS` to accumulate
  observations.

**Implication:** Single source of truth in `rvp9-core` (per L-09). Backward
adaptation runs main-thread-only between frames.

### 4.4 Quantizer tables (A5) consumed by transform pipeline (A4)

`ac_qlookup` / `dc_qlookup` tables in A5 §5 are read by the quantize
kernels invoked from the encoder core (A2's tokenize pipeline). A4's
probability-table coverage is for entropy coding; quantizer tables are
separately owned by A5 (per A5 §5).

**Implication:** Quant tables ship in `rvp9-core::quant` and are referenced
by both mode decision (cost estimates) and final tokenization.

### 4.5 Real-time identifier: `sf.recode_loop == DISALLOW_RECODE`

A1 reports this as the canonical real-time path marker. All cpu-used
6–8 configurations set this. Stage 2 LLD-arch should make this a
compile-time invariant: `rvp9-encoder` does not include the recode loop.

---

## 5. Devil's advocate disposition

A7 produced 32 challenges (`07-devils-advocate.md`). Disposition:

### ACCEPTED (16) — fix flagged in Stage 1 or queued for Stage 2

| # | Challenge | Action |
|---|---|---|
| #1, #30 | A4 truncation, missing probability tables | RESOLVED — async A4 rewrote with full pareto8 + all coef tables |
| #2 | Laplacian R-D tables not transcribed | L-42; A2 patch in Stage 2 |
| #7 | `cpi->td` aliasing — fields beyond `mb`/`counts` | D-12; Stage 2 HLD-threading produces exhaustive map |
| #11 | CONFIG_VP9_HIGHBITDEPTH leakage | D-01; HLD-arch decides |
| #13 | `setjmp` × worker threads | L-66; HLD-threading documents |
| #14 | Cyclic-refresh segmentation map ownership | L-67; HLD-threading |
| #15 | `variance_low[25]` packing | L-39; LLD-mode-decision |
| #19 | REALTIME-VBR + altref | D-13; Arthur-led scope question |
| #23 | Sub-pixel filter selection | L-48; LLD-mode-decision |
| #25 | EXT_RATECTRL WebRTC | R-06 → Arthur escalation |
| #26 | `vp9_change_config` realloc | L-70; LLD-arch |
| #28 | 4×4 transform paths in RT | L-71; LLD-transforms verify |
| #32 | VP9_COMP per-field commentary | L-72; LLD-arch expand |

### DEFERRED to Stage 2 (10) — entered in decisions-needed.md

| # | Maps to |
|---|---|
| #3 (AQ modes 1/2 unreachable) | D-05 |
| #5 (`vp9_encodedframe_overshoot` opacity) | L-19 |
| #6 (frame-parallel encoding) | D-04 |
| #8 (`adaptive_rd_thresh_row_mt`) | R-08 |
| #9 (probability table sharing) | L-09 |
| #10 (conformance vector coverage) | L-65 |
| #12 (RTCD init ordering) | R-09 |
| #17 (scene detection per-layer) | L-68 |
| #18 (SVC cross-layer drops) | L-69 |
| #29 (speed-feature condition tuples) | R-10 |

### REJECTED / accepted-as-is (6) — defensible exclusions

| # | Challenge | Why rejected |
|---|---|---|
| #4 | Denoiser SIMD bit-parity caveat | Inherited from libvpx; accept ULP tolerance in differential tests (R-03 documents) |
| #16 | "30 iterations" claim not validated | Pseudo-issue; actual iteration count is data-dependent and bounded by ref_mode_set + filter candidates, ≤ 36 |
| #20 | cpu-used 5 vs 6 boundary | Scope question; PLAN.md targets 6/7/8 by default, cpu-used 5 is optional |
| #21 | Tile-rows under SVC top-resolution | 4K SVC top-layer not in PLAN.md §2 res targets; defer to v2 |
| #24 | Compound mode at speed 6+ | Confirmed dead; drop from Stage 4 LLD |
| #27 | Recode loop dead-code | Confirmed; drop function entirely |
| #31 | Trellis-quant skip | Confirmed at cpu-used ≥ 5; drop `vp9_optimize_b` from Stage 4 |

### Net new entries created in decisions-needed.md from A7: 11

D-12, D-13, L-65, L-66, L-67, L-68, L-69, L-70, L-71, L-72, L-73.

---

## 6. Verified Stage 0 assumptions

PLAN.md's Stage 0 made implicit assumptions about libvpx. Verified
against Stage 1 findings:

| Stage 0 assumption | Status | Stage 1 evidence |
|---|---|---|
| "Real-time path largely separate from RDO" | **CONFIRMED** | `sf->use_nonrd_pick_mode = 1` at cpu-used ≥ 5 is the qualitative split; `encode_with_recode_loop` is unreached (A6 key finding, A1 §7.9) |
| "SVC adds ~30% complexity on top of base" | **MOSTLY CONFIRMED** | SVC has ~10 distinct prediction structures; A6 covers ~1000 lines of SVC-specific code beyond base. ~30% is within tolerance |
| "Threading mirrors pthread tile-worker model" | **CONFIRMED** | A6 §2-§4 documents `VPxWorker` is a thin wrapper over pthreads; tile-MT and row-MT both follow this |
| "Denoiser is ~5k LOC" | **CONFIRMED** | A5 covers `vp9_denoiser.{h,c}` (~3-4k LOC) plus integration sites; close to estimate |
| "AQ is ~3k LOC, 3 modes" | **PARTIAL MISMATCH** | A5 found AQ modes 1/2/4 guarded by `!CONFIG_REALTIME_ONLY`; only mode 3 (CYCLIC_REFRESH_AQ) is real-time-reachable. PLAN.md should be updated OR we un-gate (D-05) |
| "Frame-parallel encoding doubles complexity" | **WRONG — no frame-parallel in encoder** | A6 confirms encoder is single-frame-in-flight; only DECODER has parallel mode. PLAN.md Q4 is a v2-design question, not a libvpx-mirror question |
| "BD-rate within 0.5% achievable" | **UNCHANGED** | No Stage 1 evidence to revise this Stage 4-era target |
| "1080p60 hits real-time on commodity hardware" | **UNCHANGED** | Stage 0 baseline (5.04 ms/frame for libvpx vpxenc cpu-used 7 on testsrc2) suggests yes for synthetic; real content TBD Stage 4 |

**Material Stage 0 corrections recommended:**

1. PLAN.md §2 "Adaptive quantization (3 AQ modes)" — either un-gate
   AQ 1/2 from `CONFIG_REALTIME_ONLY` (extra LLD work) or drop to
   "AQ mode 3 (CYCLIC_REFRESH) only" (simpler).
2. PLAN.md Q4 framing — change from "Frame-parallel in v1 vs v2?" to
   "Frame-parallel needed at all in v1, given libvpx has no
   precedent?" Default: defer to v2.

---

## 7. Stage 2 entry briefing

Single-page summary for the HLD/LLD writers in Stage 2.

### HLD-arch (`notes/hld/00-architecture.md`)

**Owns:** crate boundaries, module graph, type system overview, error
model, `unsafe` policy, integer-width policy, HBD decision, build
configuration.

**Inputs:** A1 (especially §2 god-struct, §3 lifecycle), A4 §6 (RTCD
overview), 00-overview.md §3 (state graph).

**Key decisions to resolve:**
- D-01 HBD in v1
- D-08 BufferPool API exposure
- D-09 `Result` vs panic policy
- L-29 `VP9_COMP` decomposition strategy
- L-31 Worker pool eager vs lazy
- L-32 Pin to REALTIME mode
- L-33 Drop libvpx `#ifdef` knobs
- L-34 Confirm `encode_with_recode_loop` dropped
- L-37 Arc vs Rc for frame pool
- L-72 VP9_COMP per-field commentary

### HLD-threading (`notes/hld/01-threading.md`)

**Owns:** tile-MT design in Rust, `Send`/`Sync` strategy, `UnsafeCell`
policy, sync primitives, aliasing contract.

**Inputs:** A6 (especially §2 + §7), A1 §2.4 (aliasing invariants),
00-overview.md §3 (state graph), §4 (cross-cutting).

**Key decisions to resolve:**
- D-03 Threading primitives
- D-04 Frame-parallel encoding (Q4)
- D-10 `SPEED_FEATURES` snapshot vs UnsafeCell
- D-11 `MACROBLOCK` aliasing strategy
- D-12 Comprehensive `cpi->td` aliasing catalogue
- L-56 `VPxWorker` → `std::thread::scope`
- L-57 Per-worker FRAME_COUNTS accumulation
- L-66 `setjmp` × worker threads
- L-67 Cyclic-refresh segmentation map ownership
- R-08 `adaptive_rd_thresh_row_mt` non-determinism — recommend bit-
  exact-by-default

### HLD-simd (`notes/hld/02-simd-dispatch.md`)

**Owns:** RTCD-equivalent design, per-arch intrinsics organisation,
runtime feature detection, scalar-fallback policy.

**Inputs:** A1 §6 (overview), A2 §6 + A3 §6 + A4 §6 + A5 §6 + A6 §6
(kernel lists per domain).

**Key decisions to resolve:**
- D-02 RTCD: hand-written vs `build.rs`
- R-09 RTCD init-order invariant

**Aggregated kernel catalogue (from specialist §6 sections):**
- A3 (ME-heaviest): SAD/SAD-skip/SADx4D/SAD-skip-x4D, variance,
  sub-pixel variance, sub-pixel-avg variance, comp_avg_pred,
  integral projection, Hadamard 8/16/32, SATD
- A2 (encoder-core): partition variance, encode_breakout variance
- A4: scalar only (boolean coder is sequential)
- A5: denoiser kernel only (with bit-parity caveat R-03)
- All: per-arch dispatch for SSE2 / SSE4.1 / AVX2 (x86_64), NEON (aarch64)

### HLD-svc (`notes/hld/04-svc.md`)

**Owns:** SVC layer pipeline, temporal + spatial layer dependency
tracking, cross-layer rate control.

**Inputs:** A6 §2-§5 (layer contexts, ref structures verbatim),
A5 (per-layer RC interactions), A4 (layer metadata bitstream).

**Key decisions to resolve:**
- L-36 SVC reference-slot management
- L-58 SVC layer-context swap ordering
- L-61 SVC pool sizing to top-resolution
- L-68 `vp9_scene_detection_onepass` per-layer
- L-69 SVC cross-layer frame-drop interactions

### Per-module LLDs (15 total)

| LLD | Primary input | Key items |
|---|---|---|
| bitstream.md | A4 | Bool coder bit-exact contract (A4 §4.9 — 10 invariants) |
| entropy.md | A4 | Probability tables shared via rvp9-core (L-09) |
| transforms.md | A4 §5 (scan orders) + Stage 0 LLDs | L-71 verify 4×4 reachability |
| quant.md | A5 (quant tables) + A4 | L-26 vp9_compute_qdelta_by_rate |
| intra-pred.md | A2 (intra mode walk) | Stage 3 work |
| loopfilter.md | A1 (overview) | Stage 1 didn't go deep — possible Stage 1.5 re-dispatch |
| frame-buffers.md | A1 §2 + 00-overview §3 | BufferPool 12-slot design (D-08, L-37) |
| motion-estimation.md | A3 | L-05–L-08, L-40 |
| mode-decision.md | A2 | L-38–L-55 (18 items) |
| partition.md | A2 §3-§4 | L-52, L-53 |
| rate-control.md | A5 | L-01–L-04, L-16–L-26 |
| denoiser.md | A5 | R-03 (SIMD bit-parity); L-22 |
| aq.md | A5 | D-05 (mode 1/2 in v1?) |
| svc.md | A6 + L-68, L-69, L-36 | Cross-layer behaviour |
| tile-threading.md | A6 + 00-overview §3 (aliasing) | The hardest LLD; D-10, D-11, R-08 |

### Material Stage 1 follow-up

- **R-01** A4 truncation: RESOLVED by async A4 rewrite.
- **R-07** Laplacian R-D tables: verify A4 §5.7/§5.8 covers
  `rate_tab_q10`/`dist_tab_q10`. If still missing (they live in
  `vp9/encoder/vp9_rd.c`, possibly out of A4 scope), A2 patch needed.
- **#7** `cpi->td` aliasing exhaustive catalogue: Stage 2 HLD-threading
  must produce, OR re-dispatch A1+A6 with focused scope.
- **Loop filter** got light coverage (covered briefly in A1 §3; no
  dedicated specialist). Stage 1.5 re-dispatch may be needed when
  LLD-loopfilter starts.

### Stage 1 deliverable inventory

| Deliverable | Lines | Citations | Status |
|---|---|---|---|
| 01-architecture.md | 3081 | 173 | ✅ |
| 02-encoder-core.md | 2311 | 120 | ✅ |
| 03-motion-estimation.md | 1896 | 133 | ✅ |
| 04-bitstream-entropy.md | 4556 | 294 | ✅ (async rewrite) |
| 05-rate-control.md | 1979 | 175 | ✅ |
| 06-threading-svc.md | 2002 | 119 | ✅ |
| 07-devils-advocate.md | 480 | (own analysis) | ✅ (manual; async may add) |
| 00-overview.md (this doc) | ~600 | cross-refs | ✅ (manual; async may add) |
| decisions-needed.md | 489 | cross-refs | ✅ (D-13, L-73, R-10) |

**Stage 1 total content:** ~16,400 lines, 1100+ file:line citations.

---

**Stage 1 is complete pending:** (a) async A8 may replace this 00-overview
with a fuller version; (b) **Arthur's peer review and sign-off** (the
explicit human gate per PLAN.md §6 Stage 1 exit criteria).
