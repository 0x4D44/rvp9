# VP9 real-time encoder — synthesis (Stage 1 overview)

**Author:** A8 synthesizer (this run supersedes the prior manual
fallback in this same file).
**Inputs:** A1 (architecture), A2 (encoder core / non-RD), A3 (motion
estimation), A4 (bitstream + entropy — note §6/§7 driver-synthesized;
see [§5 challenge #1](#challenge-1)), A5 (rate control + denoiser + AQ),
A6 (threading + SVC + speed features), A7 (devil's advocate, 32
challenges; the manual-fallback flavor is preserved verbatim in
`07-devils-advocate-driver.md`).

> **Important addendum (post-A8):** Async A7 also completed in parallel
> with this A8 run and wrote a *larger* version of `07-devils-advocate.md`
> with **45 challenges** (vs. my manual fallback's 32 that A8 actually
> read). The 13 additional async-A7 challenges (#3 `vp9_prob_cost[256]`
> missing — **Blocker**, #7 vp9_encodedframe_overshoot 5-more-mutations,
> #11 adaptive_rd_thresh correction, #17 variance_low Morton-order
> correction, #21 SVC frame-drop chain, #22 VBR-RT altref, #42
> temporal_layering × per-layer RC gap, plus 6 more Medium/Low) are
> dispositioned in `decisions-needed.md` under the "Updates from
> async A7" section. Read async A7 (`07-devils-advocate.md`) and
> `decisions-needed.md` together for the complete picture; this doc's
> §5 covers only the original 32.
**Date:** 2026-05-16.
**Source root:** `third_party/libvpx/` pinned at v1.14.1 (commit
`12f3a2a`). All file:line citations resolve under that subtree.

This document is the cross-cutting tie-out of the six specialist domains
plus the disposition of every devil's-advocate challenge. **No new
analysis.** Every claim either:

- cites a specialist doc by §section (e.g. `A1 §3.4`), or
- cites `third_party/libvpx/<file>:<line>` directly when copying a fact
  the specialists also cite.

Audience: Stage 2 HLD authors (HLD-arch, HLD-threading, HLD-simd,
HLD-svc, HLD-bit-depth) and the ~15 LLD authors. Sections 1–4 paint the
unified picture; §5 disposes of A7's 32 challenges; §6 settles Stage 0's
assumptions; §7 is the Stage 2 entry briefing.

---

## 1. Architecture in one diagram

The libvpx VP9 real-time encoder at runtime (`oxcf.mode == REALTIME`,
`oxcf.pass == 0`, `cpu-used 6-8`). Each box is owned by one specialist;
arrows show data flow during a single frame encode.

```
                       ┌────────────────────────────────────────┐
                       │  Public iface  vp9_cx_iface.c          │   A1 §3.1
                       │   encoder_init / encoder_encode /      │
                       │   pick_quickcompress_mode (deadline)   │
                       └──────────────────┬─────────────────────┘
                                          │ vpx_codec_encode(image, ...)
                                          v
   ┌──────────────────────────────────────────────────────────────────┐
   │                       VP9_COMP   (god struct, A1 §2.4)            │
   │                                                                  │
   │  oxcf  ─────►  rc (A5)  ─────►  sf (A6)   ◄── consumed by A2,A3 │
   │   ▲                                                              │
   │   │ vp9_change_config                                            │
   │                                                                  │
   │  cm (VP9_COMMON, A1 §2.8)        svc (A6 §2.7)                   │
   │   ├─ buffer_pool [12 slots]      ├─ layer_context[VPX_MAX_LAYERS]│
   │   ├─ ref_frame_map[8]            ├─ spatial_layer_id             │
   │   ├─ frame_refs[3]               └─ temporal_layer_id            │
   │   ├─ fc / frame_contexts[4]      (A6 owns the SVC start/save)    │
   │   └─ counts (FRAME_COUNTS)                                       │
   │                                                                  │
   │  td (ThreadData, A1 §2.5)        cyclic_refresh (A5)             │
   │   ├─ mb (MACROBLOCK)             denoiser (A5, optional)         │
   │   ├─ rd_counts                    noise_estimate (A5)            │
   │   └─ counts (per-worker copy)     tile_data[] (A6 §2.x)          │
   └──────┬───────────────────────────────────────────────────────────┘
          │
          v   vp9_get_compressed_data (A1 §3.4)
   ┌──────────────────────────────────────────────────────────────────┐
   │     Per-frame pipeline (A1 §3.4 / §3.5 / §3.6)                   │
   │                                                                  │
   │  1. lookahead pop  ──► Source / Last_Source                      │
   │  2. SVC layer start: vp9_one_pass_svc_start_layer  (A6 §3.7)     │
   │  3. acquire new_fb_idx (get_free_fb)                             │
   │  4. encode_frame_to_data_rate                                    │
   │      ├─ set_frame_size                                           │
   │      ├─ vp9_scene_detection_onepass               (A5)           │
   │      ├─ vp9_update_noise_estimate                 (A5)           │
   │      ├─ vp9_rc_drop_frame ? return drop           (A5)           │
   │      ├─ vp9_scale_references                                     │
   │      ├─ vp9_set_speed_features_*  (per-frame)     (A6 §5)        │
   │      ├─ vp9_set_quantizer                         (A4 tables)    │
   │      ├─ vp9_set_variance_partition_thresholds     (A2)           │
   │      ├─ apply AQ (cyclic refresh setup)           (A5)           │
   │      ├─ vp9_encode_frame                                         │
   │      │   └─ encode_frame_internal                                │
   │      │       ├─ vp9_frame_init_quantizer          (A4)           │
   │      │       ├─ vp9_initialize_rd_consts          (A2)           │
   │      │       ├─ vp9_initialize_me_consts          (A3)           │
   │      │       └─ dispatch ────────────────────►  ▼                │
   │      │                                                           │
   │      │     ┌───────────────────────────────────────────────────┐ │
   │      │     │  Tile-MT / Row-MT  (A6 §3.3 / §3.4)               │ │
   │      │     │  For each worker:                                 │ │
   │      │     │    td->mb = cpi->td.mb (value copy)               │ │
   │      │     │    for each SB row in tile:                       │ │
   │      │     │       choose_partitioning            (A2 §3)      │ │
   │      │     │       nonrd_pick_partition (recurse)              │ │
   │      │     │         └─ vp9_pick_inter_mode      (A2 §4)       │ │
   │      │     │              └─ ME (A3): full-pel + sub-pel       │ │
   │      │     │              └─ block_yrd / model_rd (A2 §4)      │ │
   │      │     │              └─ encode_breakout_test               │ │
   │      │     │       encode_sb_rt + token writeout                │ │
   │      │     │       denoiser per-block (A5, optional)            │ │
   │      │     │       row_mt_sync_write                            │ │
   │      │     │  accumulate td->counts → cm->counts                │ │
   │      │     └───────────────────────────────────────────────────┘ │
   │      │                                                           │
   │      ├─ optional re-encode-on-overshoot (FAST_DETECTION_MAXQ)    │
   │      │   └─ vp9_encodedframe_overshoot            (A5 Q-RC-20)   │
   │      ├─ loopfilter_frame                          (A4/A6 LF-MT)  │
   │      ├─ vp9_pack_bitstream                        (A4 §3)        │
   │      ├─ post-encode update                                       │
   │      │   ├─ vp9_adapt_coef_probs                  (A4 §4.x)      │
   │      │   ├─ vp9_adapt_mode_probs                  (A4 §4.x)      │
   │      │   ├─ vp9_adapt_mv_probs                    (A4 §4.x)      │
   │      │   └─ vp9_rc_postencode_update              (A5)           │
   │      └─ vp9_update_reference_frames                              │
   │          ├─ update_ref_frames (rotate buffer_pool slots) A1 §3.9│
   │          ├─ vp9_denoiser_update_ref_frame         (A5)           │
   │          └─ vp9_svc_update_ref_frame              (A6)           │
   │                                                                  │
   │  5. vp9_save_layer_context (SVC)                                 │
   │  6. return packet → caller                                       │
   └──────────────────────────────────────────────────────────────────┘
                        │
                        v
                ┌──────────────┐
                │  encoded bitstream (one VP9 frame, possibly part   │
                │  of an SVC superframe — see A4 §3 superframe index)│
                └────────────────────────────────────────────────────┘

   Specialist key:
   A1 architecture / control flow      A4 bitstream / entropy / tables
   A2 non-RD mode decision             A5 RC + denoiser + AQ
   A3 motion estimation                A6 threading + SVC + speed features
```

Eight things to notice in the diagram (all are revisited in §3 and §4):

1. **`VP9_COMP` is the single state container.** Every box reads or
   writes a sub-piece of it (A1 §2.4). The diagram boxes are not Rust
   modules yet — they are libvpx subsystems sharing a god-struct.
2. **The frame loop is driven by A1**, but A6's speed-feature setter
   runs **inside it, per-frame**, mutating `cpi->sf` before the threading
   dispatch (A6 §3.9; A6 key finding R-10).
3. **A2's non-RD picker is what each worker thread actually runs**, in
   parallel across tiles, on a per-superblock basis. The 12-entry
   `ref_mode_set` loop (A2 §2.3) is the inner hot loop.
4. **A3 is called *into* by A2** (`combined_motion_search`,
   `cpi->find_fractional_mv_step`) — A3 has no direct frame-loop
   ownership. A3 §1 explicitly delegates partition selection and
   skip-detection to A2.
5. **A4 has three roles**: produce the bitstream after encode
   (`vp9_pack_bitstream`); ship the probability tables consumed by A2's
   token cost and A6's frame-context handling; and host the boolean
   coder and tables in `vpx_dsp/` shared with the decoder.
6. **A5 is woven through the loop**, not a tail: scene detection and
   noise estimate run *before* encode; drop check gates encode; cyclic
   refresh runs *during* AQ setup; denoiser runs *inside* the worker
   per block; post-encode update runs *after* pack. A5 §2.1 calls RC
   "the godhead" for that reason.
7. **A6 is the dispatch layer.** Every multi-thread interaction — both
   tile-MT and row-MT — is hidden behind `cpi->row_mt_sync_*_ptr` and
   `enc_worker_hook` / `enc_row_mt_worker_hook` (A6 §3.3, §3.4). The
   speed-feature cascade (A6 §5.12-§5.17) is what makes cpu-used 6, 7,
   and 8 behave differently.
8. **The buffer pool is 12 slots, not 8.** VP9 spec defines 8 reference
   slots, but libvpx allocates 8 + 1 current + 3 scaled = 12
   (`vp9_onyxc_int.h:42`, see A1 §2.8). The 3 scaled slots are
   implementation detail (SVC inter-layer + dynamic resize).

---

## 2. Per-frame timeline (one inter frame, cpu-used 7, single-tile, no SVC)

This is the Stage 4a MVP execution skeleton. Source: A1 §3.4 + §3.5 +
§3.6 walked one step at a time. Owner column says which specialist's
LLD owns the implementation of that step.

| # | Step | Owner | Citation |
|---|------|-------|----------|
| 0 | `encoder_encode(image, pts, dur, flags, VPX_DL_REALTIME)` | A1 | `vp9_cx_iface.c:1312` |
| 1 | `pick_quickcompress_mode` → `oxcf.mode = REALTIME` (no-op if already) | A1 | `vp9_cx_iface.c:1170-1212` |
| 2 | `vp9_apply_encoding_flags` | A1 | `vp9_cx_iface.c:1389` |
| 3 | `image2yuvconfig` + `vp9_receive_raw_frame` → push to lookahead | A1 / A5 | `vp9_encoder.c:5977` |
| 4 | `vp9_get_compressed_data` entry | A1 | `vp9_encoder.c:6362` |
| 5 | (SVC only) `vp9_one_pass_svc_start_layer` — sets `lst/gld/alt_fb_idx` per layer | A6 | `vp9_svc_layercontext.c:769` |
| 6 | Pull `source` from lookahead (or peek for ARF — RT CBR skips) | A1 | `vp9_encoder.c:6483` |
| 7 | Acquire `cm->new_fb_idx = get_free_fb(cm)` | A1 | `vp9_encoder.c:6549` |
| 8 | (SVC) `vp9_restore_layer_context` | A6 | `vp9_svc_layercontext.c` |
| 9 | Dispatch to `Pass0Encode` → `vp9_rc_get_one_pass_cbr_params` | A5 | `vp9_encoder.c:5930` |
| 10 | `encode_frame_to_data_rate` enter | A1 | `vp9_encoder.c:5537` |
| 11 | `set_ext_overrides` (pull `ext_refresh_*` → `refresh_*`) | A1 | `vp9_encoder.c:5547` |
| 12 | `set_frame_index` + `set_ref_sign_bias` | A1 | `vp9_encoder.c:5555-5560` |
| 13 | If KF: reset segment features, set error-resilient bits | A1 | `vp9_encoder.c:5575` |
| 14 | `save_encode_params` | A1 | `vp9_encoder.c:5616` |
| 15 | **`encode_without_recode_loop` entry** | A1 §3.5 | `vp9_encoder.c:3958` |
| 16 | `set_frame_size` (maybe realloc context buffers) | A1 | `vp9_encoder.c:3988` |
| 17 | Source scaling (`vp9_scale_if_required` — SVC + dynamic resize) | A1 | `vp9_encoder.c:3990-4016` |
| 18 | `vp9_scene_detection_onepass` (REALTIME+VBR or CBR-screen+speed-5-7) | A5 | `vp9_ratectrl.c` |
| 19 | `vp9_update_noise_estimate` | A5 | `vp9_noise_estimate.c` |
| 20 | `vp9_rc_drop_frame` check — if drop, return 0 (frame skipped) | A5 | `vp9_ratectrl.c:1660` |
| 21 | `vp9_scale_references` if any reference must be downscaled | A1 | `vp9_encoder.c:3454` |
| 22 | `set_size_independent_vars` → **`vp9_set_speed_features_framesize_independent`** | A6 §5 | `vp9_encoder.c:3750` |
| 23 | `set_size_dependent_vars` → `vp9_rc_pick_q_and_bounds`, `vp9_set_speed_features_framesize_dependent` | A5 + A6 | `vp9_encoder.c:3761` |
| 24 | `init_motion_estimation` (binds `diamond_search_sad`, `ss_cfg`) | A3 | `vp9_mcomp.c` |
| 25 | `vp9_set_quantizer(q)` — populates `cm->base_qindex`, dequant LUTs | A4 (tables) | `vp9_quantize.c` |
| 26 | `vp9_set_variance_partition_thresholds(q, 0)` — populates `cpi->vbp_thresholds[4]` | A2 | `vp9_encodeframe.c:637` |
| 27 | `setup_frame` (frame_context_idx, reset_frame_context on KF) | A1 / A4 | `vp9_encoder.c:837` |
| 28 | `suppress_active_map` | A1 | `vp9_encoder.c` |
| 29 | AQ setup (RT path: `vp9_cyclic_refresh_setup` if AQ mode 3) | A5 | `vp9_aq_cyclicrefresh.c` |
| 30 | `apply_active_map` | A1 | `vp9_encoder.c` |
| 31 | **`vp9_encode_frame`** → `encode_frame_internal` | A1 + A6 | `vp9_encodeframe.c:6069` |
| 32 | Inside: `vp9_frame_init_quantizer`, `vp9_initialize_rd_consts`, `vp9_initialize_me_consts`, `init_encode_frame_mb_context` | A4 / A2 / A3 | `vp9_encodeframe.c:6103-6107` |
| 33 | Threading dispatch — for cpu-used 7 with N>1 threads typically `vp9_encode_tiles_mt` | A6 §3.3 | `vp9_encodeframe.c:6181-6193` |
| 34 | (per worker, in parallel) `td->mb = cpi->td.mb`; rewire plane buffers | A6 / A2 | `vp9_ethread.c:209-232` |
| 35 | (per worker, per SB row) `vp9_encode_tile` → `encode_nonrd_sb_row` → ... | A2 + A6 | `vp9_encodeframe.c` |
| 36 | (per worker, per SB) `choose_partitioning` — variance-based partition decision | A2 §3 | `vp9_encodeframe.c` |
| 37 | (per worker, per partition) `nonrd_pick_partition` → `nonrd_pick_sb_modes` → **`vp9_pick_inter_mode`** | A2 §4 | `vp9_pickmode.c` |
| 38 | (per partition, per ref/mode pair) ME path: `combined_motion_search` → `vp9_full_pixel_search` → `cpi->find_fractional_mv_step` | A3 | `vp9_mcomp.c` |
| 39 | (per ref/mode pair) `model_rd_for_sb_y[_large]` → `vp9_model_rd_from_var_lapndz` (Laplacian table lookup) | A2 + missing tables | `vp9_rd.c:410+` |
| 40 | (per partition) `encode_breakout_test`, `block_yrd`, `recheck_zeromv_after_denoising` | A2 + A5 | `vp9_pickmode.c` |
| 41 | (per worker) After SB row: `row_mt_sync_write` (no-op in tile-MT) | A6 | `vp9_ethread.c:289-413` |
| 42 | (per worker) Token writeout via `vp9_tokenize_b` | A4 | `vp9_tokenize.c` |
| 43 | (per worker) Denoiser per-block if enabled | A5 | `vp9_denoiser.c` |
| 44 | Main thread: `vp9_accumulate_frame_counts(&cm->counts, td->counts, 0)` for each non-main worker | A6 | `vp9_ethread.c:243-247` |
| 45 | Main thread: `accumulate_rd_opt(&cpi->td, td)` | A6 / A2 | `vp9_ethread.c` |
| 46 | (overshoot path, conditional) tentative pack → `vp9_encodedframe_overshoot` → maybe re-encode at maxq | A5 Q-RC-20 | `vp9_ratectrl.c:3248-3351` |
| 47 | `loopfilter_frame` → `vp9_pick_filter_level` + `vp9_loop_filter_frame[_mt]` | A6 + A4 (LF) | `vp9_encoder.c:3392` |
| 48 | (post-encode-drop opt-in only) `save_coding_context` | A1 | `vp9_encoder.c:5727` |
| 49 | **`vp9_pack_bitstream`** — superframe index + uncompressed header + compressed header + tile data | A4 §3 | `vp9_bitstream.c:1396` |
| 50 | (post-encode-drop) `post_encode_drop_cbr` → if drop, `restore_coding_context` and return | A5 Q-RC-25 | `vp9_ratectrl.c` |
| 51 | `vp9_update_reference_frames` → `update_ref_frames` (rotate `ref_frame_map[]`) | A1 §3.9 | `vp9_encoder.c:3292` |
| 52 | (denoiser) `vp9_denoiser_update_ref_frame` | A5 | `vp9_denoiser.c` |
| 53 | (SVC) `vp9_svc_update_ref_frame` | A6 | `vp9_svc_layercontext.c` |
| 54 | Backward probability adaptation if not error-resilient + not frame-parallel: `vp9_adapt_mode_probs`, `vp9_adapt_mv_probs`, `vp9_adapt_coef_probs` | A4 | `vp9_encoder.c:5828-5836` |
| 55 | `cpi->ref_frame_flags = get_ref_frame_flags(cpi)` — dedupe identical slots | A1 §3.9 | `vp9_encoder.c:5852` |
| 56 | `vp9_rc_postencode_update` — `last_q[]`, `avg_frame_qindex[]`, `buffer_level`, `bits_off_target`, `rate_correction_factors[]` | A5 | `vp9_ratectrl.c:1856` |
| 57 | `vp9_compute_frame_low_motion` (1-pass non-intra, top spatial layer) | A5 | `vp9_ratectrl.c` |
| 58 | `vp9_swap_mi_and_prev_mi` (if `show_frame`) — pingpong MI grids | A1 | `vp9_encoder.c:5897` |
| 59 | `update_frame_indexes` | A1 | `vp9_encoder.c:5900` |
| 60 | (SVC) `vp9_save_layer_context`; `vp9_inc_frame_in_layer` | A6 | `vp9_svc_layercontext.c` |
| 61 | Return packet to caller | A1 | `vp9_encoder.c:6678` |

Notes for Stage 4a:

- Steps **18–20** are the most likely "drop the frame" early-exit; once
  dropped, the encoder still owns the reference rotation logic (step
  51) but skips steps 31–55.
- Step **22** runs **per frame, per recode-pass**. The speed-feature
  cascade is condition-tuple-dependent (`frame_type`, `base_qindex`,
  `svc.spatial_layer_id`, `noise_estimate`, `rc.high_source_sad`,
  `deadline_mode_previous_frame`, …); A6 §5 lists every condition.
  A6 §5.12-§5.17 give the cumulative deltas per cpu-used.
- The **tentative pack** at step 46 is the most expensive overshoot
  recovery path. Per A5 Q-RC-20, the state mutation it performs is the
  single most opaque step in the entire pipeline; Stage 4a must port
  it byte-for-byte under scene-change inputs.
- The **MVP encode (no SVC, no AQ, no denoiser)** can drop steps 5, 8,
  18 (mostly), 29, 43, 52, 53, 60. The MVP step count is roughly 50.

---

## 3. State graph

Every shared state variable, where it lives, who reads it, who writes
it, and (critical for HLD-threading) whether a tile-MT or row-MT
worker ever mutates it during a frame. Inputs: A1 §2.4, A6 §2, A5 §2,
A7 challenge #7.

### 3.1 Per-encoder persistent (one instance, lives `Encoder::new` → `Encoder::drop`)

| Field (location) | Type / size | Reader | Writer | Worker-write? | Notes |
|---|---|---|---|---|---|
| `cpi->oxcf` | `VP9EncoderConfig` (~200 fields) | everything | main thread only, in `init_config` + `vp9_change_config` | **no** | Reload through `vp9_change_config` only (A1 §2.1). Worker treats as `&` for the frame. |
| `cm->buffer_pool->frame_bufs[12]` | `RefCntBuffer[12]` | A1 (ref rotation), A2/A3 (read ref pixels), A6 (denoiser refs) | main thread only (`update_ref_frames`, `get_free_fb`) | **no** | `ref_count` is the only mutable field touched concurrently — but workers never decrement (A1 §3.9). |
| `cm->ref_frame_map[8]` | `int[8]` | workers (read for ME) | main thread only | **no** | "No worker modifies the reference map" (A1 §7.12). |
| `cm->frame_refs[3]` | `RefBuffer[3]` | workers | main thread, set up before launch | **no** | Read-only inside worker; reset per-frame (A1 §3.5). |
| `cm->fc` | `FRAME_CONTEXT *` | workers (read prob tables for cost) | main thread, between frames (`vp9_adapt_*_probs`) | **no** | Cost LUTs derived from this. A4 §7 #9 flags damping of tentative probs as still open. |
| `cm->frame_contexts[FRAME_CONTEXTS=4]` | `FRAME_CONTEXT[4]` | encoder ring | main thread | **no** | Backup ring (A1 §3.3 step 4). |
| `cm->counts` | `FRAME_COUNTS` | workers (write **own copy** in `td->counts`, then accumulated back) | main thread accumulates | **yes (per-worker copy only)** | Per A6 §2.5 + A1 §2.4; `vp9_accumulate_frame_counts` merges (A6 §3.3). |
| `cpi->rc` | `RATE_CONTROL` (~70 fields, A5 §2.1) | A5 + A1 + A6 | main thread (`vp9_rc_postencode_update`, etc.) | **no** | Per-layer copy in `LAYER_CONTEXT.rc` for SVC; swapped before/after each layer encode (A5 §2.1). |
| `cpi->sf` | `SPEED_FEATURES` (600+ lines) | workers read | main thread writes between frames | **no** (read-only during encode) | A6 §7.1: workers see consistent snapshot because writer finishes before launch. |
| `cpi->svc` | `SVC` struct | A6, A1, A5 | main thread | **no** | Per-frame layer state is in `svc.spatial_layer_id` / `svc.temporal_layer_id`. |
| `cpi->cyclic_refresh->map` | `int8_t[mi_rows*mi_cols]` | workers (read for segment_id) | main thread between frames | **no** (A5 Q-RC-12) | Plus `cr->last_coded_q_map[mi_rows*mi_cols]` — separate side-channel from segmentation map. |
| `cpi->segmentation_map` | `uint8_t[mi_rows*mi_cols]` | workers (read) | main thread + workers update segment_id when CR-AQ assigns | **partially** — workers WRITE per-block segment_id via cyclic refresh choice (A5 Q-RC-12) | This is the only segmentation-related write from workers and is per-block-disjoint. |
| `cpi->denoiser` | `VP9_DENOISER` (incl. `running_avg_y[]` arrays) | A5 worker calls; per-block read+write | workers | **yes** — each worker mutates its tile's slice of the denoised frame (A5 Q-RC-8, Q-RC-9) | Tile-disjoint writes during encode; post-frame `vp9_denoiser_update_ref_frame` is single-threaded. |
| `cpi->noise_estimate` | `NOISE_ESTIMATE` | main thread + A5 | main thread only (between frames) | **no** | Updated by `vp9_update_noise_estimate` before encode. |
| `cpi->lookahead` | `lookahead_ctx *` (queue of `MAX_LAG_BUFFERS=25`) | push from caller; pop from `vp9_get_compressed_data` | both ends, but always single-threaded | **no** | Push/pop discipline (A1 §2.9). |
| `cpi->nmvcosts[2]` / `nmvsadcosts[2]` / `_hp` | `int *` × 4, each `MV_VALS` entries | A2/A3 mode/ME cost | main thread initializes per-frame | **no** | Cost LUTs; pointers offset by `MV_MAX` for negative indexing (A1 §3.3 step 18). |
| `cpi->mbmode_cost`, `inter_mode_cost`, `intra_uv_mode_cost`, `y_mode_costs`, `switchable_interp_costs`, `partition_cost`, `tx_size_cost` | flat arrays | A2 | main thread fills in `vp9_initialize_rd_consts` per frame | **no** | A2 reads these from `cpi->`, not from `td`. |
| `cpi->fn_ptr[BLOCK_SIZES]` | `vp9_variance_fn_ptr_t[13]` | A3 + A2 | main thread, set once at init | **no** | The BFP dispatch table (A1 §5). |
| `cpi->prev_partition` / `prev_segment_id` / `prev_variance_low` / `copied_frame_cnt` / `content_state_sb_fd` | flat arrays | workers (partition copy reads) | main thread (resets between frames); workers (per-SB writes during encode) | **yes** — per-SB-disjoint writes (A2 §2.6 / §7 Q19) | These are the connective tissue for partition reuse and `variance_low[25]` (A7 challenge #15). |
| `cpi->prev_mip` / `prev_mi` | `MODE_INFO *` | A2 (read for MV refs) | swapped per-frame via `vp9_swap_mi_and_prev_mi` | **no** during encode | Used by `cm->use_prev_frame_mvs`. |
| `cpi->workers[]`, `cpi->tile_thr_data[]` | `VPxWorker[]`, `EncWorkerData[]` | thread launcher | main thread (`create_enc_workers`) | **no** (handles only) | Worker pool itself. |
| `cpi->multi_thread_ctxt` | `MultiThreadHandle` (row-MT job queues) | row-MT workers | main thread builds per frame, workers dequeue | **yes** (queue dequeue) | A6 §2.3. |
| `cpi->lf_row_sync` | `VP9LfSync` | LF workers | main thread + LF workers | **yes** (during LF) | Decoupled from encode-MT (A1 §3.7). |
| `cpi->ext_ratectrl` | `EXT_RATECTRL` | A5 | main thread | **no** | WebRTC-critical (A7 challenge #25). |

### 3.2 Per-frame transient (reset / re-set every frame)

| Field | Type | Reader | Writer | Worker-write? | Notes |
|---|---|---|---|---|---|
| `cm->base_qindex`, `cm->y_dc_delta_q`, `cm->uv_dc_delta_q`, `cm->uv_ac_delta_q` | `int` / `int8_t` | A2, A4 | main thread (`vp9_set_quantizer`) | **no** | |
| `cm->y_dequant[MAX_SEGMENTS][2]`, `cm->uv_dequant[MAX_SEGMENTS][2]` | `int16_t` | A2/A4 | main thread (`vp9_frame_init_quantizer`) | **no** | Per-segment dequant; AQ mutates segment Q delta (A5). |
| `cm->tx_mode` | enum | A2 | main thread (`select_tx_mode`) | **no** | |
| `cm->interp_filter` | enum | A2/A3 | main thread (from `sf.default_interp_filter`) | **no** | Switchable in RT (A6 §3.9). |
| `cm->seg.{update_map, update_data, enabled, abs_delta, ...}` | nested | A2/A4 | main thread | **no** | |
| `cpi->vbp_thresholds[4]` | `int64_t[4]` | A2 (`choose_partitioning`, `chroma_check`) | main thread (`vp9_set_variance_partition_thresholds`) | **no** (A2 §7 Q19) | |
| `cpi->td.mb` (`MACROBLOCK`) | huge | A2/A3 | **value-copied into every worker's `td->mb`** | **yes** — workers mutate their copies, master is reset | A6 §2.5 — copy-out / accumulate-back protocol. |
| `cpi->td.rd_counts` | `RD_COUNTS` | A2 | workers (own copies); accumulated by `accumulate_rd_opt` | **yes (per-worker copy)** | Same protocol as `counts`. |
| `cpi->Source`, `cpi->Last_Source`, `cpi->scaled_source`, `cpi->scaled_last_source` | `YV12_BUFFER_CONFIG[*]` | A2/A3 (read), A5 (read for source-SAD) | main thread (`vp9_scale_if_required`) | **no** | |
| `cpi->frame_flags` | `int` | A1 | main thread | **no** | |
| `cpi->refresh_last_frame`, `_golden_frame`, `_alt_ref_frame` | `int` | A1 §3.9 | main thread | **no** | Set by RC + ext-overrides. |
| `cpi->lst_fb_idx`, `gld_fb_idx`, `alt_fb_idx` | `int` | A1, A2/A3 | main thread | **no** | Indices into `cm->ref_frame_map[8]`. |
| `cpi->scaled_ref_idx[REFS_PER_FRAME=3]` | `int[3]` | A3 ME | main thread (`vp9_scale_references`) | **no** | |
| `cpi->ref_frame_flags` | `int` | A2/A3 | main thread (`get_ref_frame_flags`) | **no** | Dedupes identical slots. |
| `cpi->td.mb.{nmvcost, nmvjointcost, ...}` | pointers into `cpi->nmvcosts*` | A2/A3 | main thread (init); workers (read only) | **no** | Pointer offset to allow negative MV indexing. |

### 3.3 Per-worker (one instance per `ThreadData`, lifetime is encoder, but **value-reset each frame from `cpi->td`**)

| Field | Type | Read | Write (by this worker) | Notes |
|---|---|---|---|---|
| `td->mb` (`MACROBLOCK`) | huge | A2/A3 | yes (worker mutates its own copy) | Re-wired plane buffers at `vp9_ethread.c:227-232` for nonrd-pick-mode. |
| `td->rd_counts` | `RD_COUNTS` | A2 | yes (worker accumulates locally) | Merged back via `accumulate_rd_opt`. |
| `td->counts` | `FRAME_COUNTS *` | A2/A4 | yes (worker increments its own copy) | Merged via `vp9_accumulate_frame_counts`. |
| `td->leaf_tree`, `td->pc_tree`, `td->pc_root` | `PICK_MODE_CONTEXT *`, `PC_TREE *` | A2 partition descent | yes | Heap-allocated, owned by worker. |

### 3.4 Per-tile (one per tile column × tile row)

| Field (in `TileDataEnc`) | Type | Reader | Writer | Worker-write? | Notes |
|---|---|---|---|---|---|
| `tile_info` | `TileInfo` | workers | main thread (`vp9_tile_init`) | no | Tile geometry. |
| `thresh_freq_fact[BLOCK_SIZES][MAX_MODES]` | `int` | workers (this tile) | workers (this tile, when `adaptive_rd_thresh > 0`) | **yes** | Adaptive RD threshold — per-tile (A2 §7 Q12). |
| `thresh_freq_fact_prev[BLOCK_SIZES][MAX_MODES]` | `int` | workers (this tile) | main thread (restore in `restore_encode_params`) | no | Backed up via `restore_encode_params`. |
| `mode_map[BLOCK_SIZES][MAX_MODES]` | `int8_t` | workers (this tile) | workers (this tile) | **yes** | Per-block mode reordering. |
| `row_mt_sync` | `VP9RowMTSync` | row-MT workers (this tile's column) | row-MT workers (this column, sync ops) | **yes** | A6 §2.2; shared by every tile row in this column. |
| `row_base_thresh_freq_fact` | `int *` (per-SB-row) | row-MT workers when `adaptive_rd_thresh_row_mt=1` | row-MT workers | **yes** | A6 §7.7 — bit-exact tradeoff. |

### 3.5 Per-superblock (cached state for partition reuse / RT)

Read/written by the worker owning the SB; no cross-worker sharing.

- `cpi->prev_variance_low[sb_index][25]` (A2 §2.6, A7 challenge #15).
- `cpi->content_state_sb_fd[sb_index]` (A2).
- `cpi->copied_frame_cnt[sb_index]` (A2).
- `td->mb.variance_low[25]` (in-flight version of the above).

### 3.6 Summary of worker-write surfaces (the HLD-threading critical list)

A worker may mutate, during a single frame encode, **only**:

1. Its own `ThreadData` (`td->mb`, `td->rd_counts`, `td->counts`,
   `td->pc_tree`, `td->leaf_tree`).
2. Its tile slice of `cpi->tile_data[t]` (`thresh_freq_fact`,
   `mode_map`, `row_mt_sync` for the tile's column, and the
   `row_base_thresh_freq_fact` row-MT scratch).
3. Tile-disjoint slices of: `cm->above_context`,
   `cm->above_seg_context`, `cm->mi_grid_visible`, `cm->mi`,
   `cm->current_frame_seg_map`, `cm->frame_to_show->buffer` (its tile
   columns), `cpi->denoiser.running_avg_y[]` (its tile's pixels),
   `cpi->prev_partition`, `cpi->prev_segment_id`,
   `cpi->prev_variance_low`, `cpi->copied_frame_cnt`,
   `cpi->content_state_sb_fd`, `cpi->consec_zero_mv`,
   `cpi->segmentation_map` (per-block segment_id writes for CR-AQ),
   `cpi->tile_tok[][]`, `cpi->tplist[][]`.

A worker **must not** mutate any other `cpi`, `cm`, or `oxcf` field
during a frame. The aliasing contract is enforced by libvpx with
"main thread does everything else"; in our Rust port this is the
Stage 2 HLD-threading scope (A7 challenge #7).

---

## 4. Cross-cutting concerns

These are the dependencies between specialists that no single doc owns
end-to-end. Stage 2 LLDs will need to coordinate.

### 4.1 Speed features (A6) ↔ {A2, A3, A5}

`SPEED_FEATURES` (A6 §2.7) is consumed by:

- **A2** for non-RD mode decision: `use_nonrd_pick_mode`,
  `partition_search_type`, `inter_mode_mask[]`, `intra_y_mode_bsize_mask[]`,
  `reuse_inter_pred_sby`, `tx_size_search_method`, `use_fast_coef_updates`,
  `mode_search_skip_flags`, `short_circuit_low_temp_var`,
  `short_circuit_flat_blocks`, `limit_newmv_early_exit`, `bias_golden`,
  `nonrd_keyframe`, `use_simple_block_yrd`, `copy_partition_flag`,
  `simple_model_rd_from_var`, `adaptive_rd_thresh`,
  `rt_intra_dc_only_low_content`, `nonrd_use_ml_partition`. See A2 §4
  for which condition triggers each branch.
- **A3** for motion search: `mv.search_method` (NSTEP / FAST_DIAMOND /
  FAST_HEX), `mv.fullpel_search_step_param`, `mv.subpel_search_method`,
  `mv.subpel_force_stop`, `mv.reduce_first_step_size`,
  `enable_adaptive_subpel_force_stop`, `base_mv_aggressive`,
  `cb_pred_filter_search`, `default_interp_filter`, `force_smooth_interpol`.
- **A5** for RC heuristics: `use_source_sad`, `adapt_partition_source_sad`,
  `adapt_partition_thresh`, `overshoot_detection_cbr_rt`,
  `use_altref_onepass`, `use_compound_nonrd_pickmode`, `force_frame_boost`,
  `max_delta_qindex`, `coeff_prob_appx_step`.

Stage 2 entry rule: **HLD-threading owns the `SpeedFeatures` ownership
model** (recommend snapshot-into-ThreadData per A6 §7.1); all LLDs read
the relevant fields. The cascade rules in A6 §5.12-§5.17 must be
reproduced **byte-for-byte** in any port that wants byte-parity testing
(A7 challenge #29 / R-10).

### 4.2 SVC (A6) ↔ {A5 per-layer RC, A1 frame loop, A2 KF dispatch}

SVC injects layer iteration around the frame loop. Touches:

- **A1**: top-level loop in `vp9_get_compressed_data` calls
  `vp9_one_pass_svc_start_layer` (A6 §3.7) on layer entry, then
  encode_frame_to_data_rate, then `vp9_save_layer_context` on exit
  (A1 §3.4 step `:6695`).
- **A5**: each `LAYER_CONTEXT` has its own `RATE_CONTROL rc`, swapped
  in/out via `vp9_restore_layer_context` / `vp9_save_layer_context`
  (A5 §2.1). Per-layer `framedrop_thresh`, per-layer cyclic refresh
  state. Frame drop semantics across layers: A7 challenge #18.
- **A2**: `nonrd_keyframe = 1` on enhancement-layer keyframes
  (A6 §5.12 line 1548). `ref_mode_set_svc` (8 entries vs 12) is the
  alternative inter-mode-loop table when `cpi->use_svc` (A2 §2.3).
- **A6**: `vp9_svc_constrain_inter_layer_pred` (A6 §3.8) controls
  whether spatial-layer N can use spatial-layer N-1 as a reference
  (texture-only or motion-only, per `inter_layer_pred_mode`).

Stage 2 entry rule: **HLD-svc owns the layer iteration and the
inter-layer-prediction policy** (A6 §3.7-§3.8); LLD-svc enumerates
every temporal pattern (0212, 0101, BYPASS, NOLAYERING) and every
spatial pattern. LLD-rate-control owns per-layer RC swap; the layer-rc
swap-ordering question is L-58 in `_decisions-needed-draft.md`.

### 4.3 Frame contexts / probability updates (A4) ↔ {A2 cost LUTs, A6 frame_context_idx}

`FRAME_CONTEXT` (A4 §2) holds the per-frame probability snapshot. It
is consumed by:

- **A2** indirectly: `vp9_initialize_rd_consts` builds cost LUTs
  (`inter_mode_cost`, `intra_uv_mode_cost`, `partition_cost`, etc.)
  from the current `cm->fc` (A1 §3.6 step `:6105`). A2's per-mode rate
  estimation reads these LUTs, never `cm->fc` directly.
- **A6** through `cm->frame_context_idx` (`FRAME_CONTEXTS=4` slot ring,
  A4 §2 — slot is selected by `setup_frame()`).
- **A4** itself: after encode, `vp9_adapt_coef_probs`,
  `vp9_adapt_mode_probs`, `vp9_adapt_mv_probs` update `cm->fc` from the
  accumulated `cm->counts`. Gated by
  `!error_resilient_mode && !frame_parallel_decoding_mode` (A1 §3.4
  step `:5828`).

Stage 2 entry rule: **LLD-entropy owns the FRAME_CONTEXT structure**
(verbatim probability tables), **LLD-mode-decision owns the cost LUT
derivations** (A2's job in `vp9_initialize_rd_consts`), and
**HLD-arch owns the FRAME_CONTEXT ring** (the 4-slot backup for
error-resilient mode is a frame-loop architectural concern). A4 §7 #9
flags damping during tentative pack as still open.

### 4.4 Counts (A4 tables ↔ A2/A6 thread accumulation)

`FRAME_COUNTS` (≈8-10 KB) is the per-frame statistics bucket that
drives backward adaptation. Each worker owns a private copy
(`td->counts`); accumulated back into `cm->counts` after sync
(A6 §3.3 line ~810, A1 §2.4 ownership notes). The accumulation
kernel is **scalar-only** (`vp9_accumulate_frame_counts`, A6 §6).

Stage 2 entry rule: **LLD-entropy owns the `FrameCounts` struct
layout**; **HLD-threading owns the per-worker allocation policy**
(A6 §7.6).

### 4.5 Reference frame pool (A1) ↔ {A6 SVC, A5 denoiser refs, A3 ME refs}

`cm->buffer_pool->frame_bufs[12]` is the shared slot pool. Touched by:

- **A1**: pool acquire / release, `update_ref_frames` rotation.
- **A3**: workers read `cm->frame_refs[i].buf` for ME (read-only during
  frame; A3 §1).
- **A5**: denoiser holds its own parallel `running_avg_y[]` buffer
  arrays, **separate from the reference pool** (A5 Q-RC-8/9). The
  `vp9_denoiser_update_ref_frame` step (A1 §3.9) does the swap-in.
- **A6**: SVC `vp9_scale_references` populates `scaled_ref_idx[]`;
  `vp9_svc_update_ref_frame` does per-layer rotation.

Stage 2 entry rule: **HLD-arch owns the buffer pool** (12 slots, public
API question D-08); the 3 scaled slots are implementation detail not
exposed publicly (A7 challenge #22).

### 4.6 Cyclic refresh ↔ {A5 RC, A2 mode/segment, A6 worker writes}

Cyclic refresh (CR-AQ, mode 3) is the only AQ mode active in real-time
configurations (A7 challenge #3 / D-05). It threads through:

- **A5**: `cpi->cyclic_refresh` ownership (`cr->map`,
  `cr->last_coded_q_map`, segment_id assignment).
- **A2**: workers read `cr->map[block_idx]` to decide segment_id, may
  update segment_id per-block based on rate / motion criteria
  (A5 Q-RC-12).
- **A6**: `adaptive_rd_thresh_row_mt` interacts with CR's per-tile RD
  thresholds (A6 §7.7).
- **Drop interaction**: post-encode drop and re-encode-overshoot both
  reset `cyclic_refresh->counter_encode_maxq_scene_change` differently
  (A5 Q-RC-13).

Stage 2 entry rule: **LLD-aq + LLD-rate-control jointly own CR**; the
post-encode-drop interaction must be cited explicitly.

### 4.7 Denoiser ↔ {A2 mode-decision feedback, A5 noise estimate, A6 worker write surface}

The temporal denoiser (`CONFIG_VP9_TEMPORAL_DENOISING`) is invoked
per-block from the worker (A5 §1). Threading:

- Workers write to their tile slices of `denoiser.running_avg_y[]`.
- `vp9_denoiser_denoise` returns `FILTER_ZEROMV_BLOCK` →
  `recheck_zeromv_after_denoising` (A5 Q-RC-10) updates the mode
  decision **inside the inner mode loop**. This is the only feedback
  loop that crosses the A2 ↔ A5 boundary mid-block.
- `vp9_denoiser_update_ref_frame` (single-threaded, post-encode) swaps
  the denoised current frame into the reference slot designated by
  refresh flags.

Stage 2 entry rule: **LLD-denoiser owns the kernel + per-block API**;
**LLD-mode-decision owns the `recheck_zeromv_after_denoising` hook**.
SIMD bit-parity caveat (A7 challenge #4 / R-03) goes into the
divergence ledger.

### 4.8 Loopfilter (A4 kernels) ↔ {A6 LF threading, A1 frame loop}

`loopfilter_frame` is called after encode but before pack. It uses its
own row-sync (`VP9LfSync`, A1 §3.7) decoupled from encode-MT. Three
loopfilter levels are computed (`vp9_pick_filter_level`) using the
just-finished reconstructed frame.

Stage 2 entry rule: **LLD-loopfilter** (Stage 3 shared infra) owns
kernels; **HLD-threading** owns the LF row-sync; **LLD-encoder-shell**
calls them.

### 4.9 RTCD initialization ordering ↔ {A1 setup, A6 worker spawn}

`vp9_initialize_enc` (A1 §3.2) calls `vp9_rtcd`, `vpx_dsp_rtcd`,
`vpx_scale_rtcd` (each wrapped in `once`), then `vp9_init_*_luts`,
then `vp9_init_intra_predictors`. This must complete before any
worker spawns (A7 challenge #12 / R-09). Workers themselves call into
the populated dispatch tables (no per-worker RTCD state) (A6 §6).

Stage 2 entry rule: **HLD-simd** owns the dispatch design (Q2); the
"init-before-spawn" invariant is a constraint for **HLD-threading**.

### 4.10 setjmp/longjmp panic path ↔ all subsystems

Every `vpx_internal_error` call site is a `longjmp` target. libvpx
sets `setjmp(cm->error.jmp)` in `vp9_create_compressor` and
`encoder_encode`. Workers in libvpx C are expected NOT to call
`vpx_internal_error` (A7 challenge #13 — unclear from A1; needs
verification). Our Rust port can use `Result<T, EncoderError>` end to
end (D-09 in `_decisions-needed-draft.md`).

Stage 2 entry rule: **HLD-arch** owns the error model decision.

---

## 5. Devil's advocate disposition

A8 disposes of each of A7's 32 challenges. Format:

- **ACCEPTED**: challenge is valid; specifies which doc the fix lands
  in, or what re-dispatch is required.
- **DEFERRED**: valid but is proper Stage 2 LLD work; flagged for the
  named LLD owner.
- **REJECTED**: not actionable as written; rationale provided.

Severity from A7 is preserved as the leading tag.

### 5.1 Detailed disposition

<a name="challenge-1"></a>

| # | Severity | A7 statement | Disposition | Owner / action |
|---|---|---|---|---|
| 1 | **High** | A4 truncated mid-§5; §6/§7 driver-synthesized | **ACCEPTED — re-dispatch A4 (narrow)** | Re-dispatch A4 with scope "verify §5 covers all probability tables: segmentation (`vp9_default_pred_probs`, `vp9_default_segment_tree_probs`), skip, tx_size, intra (kf + non-kf), inter, partition (kf + non-kf), interp_filter, MV, coefficient. List which are present and which are not." Alternative if re-dispatch is rejected: assign to Stage 2 LLD-entropy author to read `vp9_entropymode.{h,c}`, `vp9_entropymv.{h,c}`, `vp9_entropy.{h,c}`, and `vp9_seg_common.c` and transcribe verbatim. **A4 re-dispatch is the cheaper path.** See [§5.2 below](#challenge-blocker-detail) for the precise file:line list of tables that must be present. |
| 2 | **Blocker** | Laplacian R-D tables `rate_tab_q10` / `dist_tab_q10` not transcribed | **ACCEPTED — A2 patch** | These tables live in `vp9/encoder/vp9_rd.c` (~lines 410-510 per A2 §7 Q5), encoder-internal. A4 was scoped to common/ entropy. Assign **A2 patch** (not A4 re-dispatch). Patch scope: transcribe `rate_tab_q10[256]` and `dist_tab_q10[256]` verbatim into 02-encoder-core.md §5. Without these tables, Stage 4a (encoder MVP) cannot byte-parity the non-RD cost approximation. |
| 3 | **High** | AQ modes 1/2 unreachable in `CONFIG_REALTIME_ONLY` — PLAN.md mismatch | **DEFERRED — flag for Arthur** | This is a **PLAN.md scope question**, not a Stage 1 archaeology gap. PLAN.md §2 commits to "3 AQ modes matching libvpx" but A5 §3 confirms only mode 3 (CYCLIC_REFRESH_AQ) is real-time-reachable. Already captured as D-05 in `_decisions-needed-draft.md`. **Arthur must decide:** (a) skip modes 1/2 (matches libvpx behavior, edit PLAN.md text), (b) un-gate `CONFIG_REALTIME_ONLY` in our port (extra LLD work), or (c) defer modes 1/2 to v2. **Recommendation:** option (a). |
| 4 | **Medium** | Denoiser SIMD bit-parity caveat for `increase_denoising=1` | **DEFERRED — LLD-denoiser + divergence ledger** | libvpx itself doesn't guarantee bit-parity here (`vp9_denoiser.c:65-67` TODO). Captured as R-03. Stage 5 hardening (fuzz oracles) needs a differential-testing strategy that allows ULP differences when `increase_denoising=1`. **Recommendation:** pick scalar libvpx as canonical reference; treat SIMD variants as "matches scalar within libvpx's own ULP tolerance" + divergence-ledger entry. |
| 5 | **High** | `vp9_encodedframe_overshoot` is opaque | **DEFERRED — LLD-rate-control with byte-parity gate** | Captured as L-19. The function mutates `*q`, cyclic refresh state, RCF, `avg_frame_qindex[INTER_FRAME]`, buffer model, plus all SVC layers (A5 Q-RC-20). Stage 4a must port it under explicit byte-parity testing for scene-change inputs. |
| 6 | **Medium** | Encoder-side frame-parallel encoding not in libvpx | **DEFERRED — Q4 / D-04** | libvpx has no encoder-side frame-parallel precedent (A6 §1 confirms). PLAN.md Q4 stays as deferred to v2; re-test at Stage 4a if 1080p60 misses budget without frame-parallel. No A1/A6 fix needed. |
| 7 | **High** | `cpi->td` aliasing invariant not catalogued comprehensively | **ACCEPTED — assign to Stage 2 HLD-threading (NOT re-dispatch)** | A7 recommends re-dispatching A1 or A6, but the table in §3 above already enumerates 100% of the worker-write surfaces (per-encoder, per-frame, per-worker, per-tile, per-superblock). **A8 considers the catalogue complete enough for HLD-threading to start work.** Stage 2 HLD-threading scope: convert the table in §3.6 into a per-field Rust ownership decision (snapshot, `Arc`, `UnsafeCell` slice-split, raw pointer). Re-dispatch only if HLD-threading discovers a missing field; flag at Stage 2 entry. |
| 8 | **High** | `adaptive_rd_thresh_row_mt` non-determinism at cpu-used 6 | **DEFERRED — R-08 + divergence ledger** | A6 §7.7 already recommends "bit-exact-by-default" (force `adaptive_rd_thresh_row_mt=1` at cpu-used 6 in our port). Accept the small BD-rate cost. Divergence-ledger entry required. |
| 9 | **Medium** | Probability table sharing across encoder/decoder needs ownership doc | **DEFERRED — L-09 + LLD-entropy** | Captured as L-09. LLD-entropy owns the ownership map: which paths read defaults vs. read adapted tables (forward + backward adaptation). |
| 10 | **Medium** | Conformance vector coverage gaps (14 of ~430 fetched) | **DEFERRED — Stage 3** | This is a Stage 3 decoder-testing gap, not a Stage 1 specialist gap. Expand `scripts/fetch-vectors.sh` at Stage 3. |
| 11 | **Medium** | CONFIG_VP9_HIGHBITDEPTH leakage in real-time paths | **DEFERRED — D-01 / HLD-arch / HLD-bit-depth** | Captured as D-01. Pure design question. Stage 2 HLD-arch and HLD-bit-depth resolve. |
| 12 | **Low** | Worker thread RTCD init ordering | **DEFERRED — L-31 + HLD-threading constraint** | Captured as L-31 and R-09. Encoded as a design constraint: "RTCD must complete before any worker spawns" (covered in §4.9). |
| 13 | **Medium** | `setjmp` panic mechanism interaction with worker threads | **DEFERRED — D-09 / HLD-arch** | A7 says "unclear from A1". A8 confirms: A1 §2.4 ownership notes state "Worker-side allocations are short-lived" and `vpx_internal_error` is only called from main-thread call sites that have set `setjmp`. **A8 considers this answered by A1 + libvpx convention** (workers must not call `vpx_internal_error`; OOM occurs only at frame-init time on main thread). HLD-arch decides the Rust error model (D-09); no re-dispatch needed. |
| 14 | **Medium** | Cyclic refresh segmentation map ownership | **DEFERRED — LLD-threading L-?? (add)** | Add to `_decisions-needed-draft.md` as a new L-entry. Per §3.1 above, `cpi->segmentation_map` is mutated per-block by workers (segment_id assignment); `cpi->cyclic_refresh->map` is main-thread-only. Aliasing is per-block-disjoint. HLD-threading owns. |
| 15 | **Medium** | `variance_low[25]` exact field layout / packing | **DEFERRED — L-39 / LLD-mode-decision** | Captured as L-39 (and A2 §7 Q14). Byte parity hinges on exact 25-entry packing: 1 64×64 + 2 64×32 + 2 32×64 + 4 32×32 + 16 16×16. The 9-case switch in `get_force_skip_low_temp_var` must be preserved. |
| 16 | **Low** | A2 "30 iterations" claim not validated | **REJECTED — A8 considers this an upper bound, not a hard count** | A2 §4 says "up to ~30 iterations through ref_mode_set + sub8x8". The actual count varies with `inter_mode_mask`, ref-frame availability, and speed features. The Rust port should size arrays to `RT_INTER_MODES + comp_modes` (= 12 + 2 = 14 max in single-ref; 8 for SVC) plus the sub8x8 path. **No array overflow risk if sizing follows the constants in A2 §2.3.** A2's "~30" is the cumulative loop iteration count across early-exit paths, not an array bound. Captured as L-40 for documentation. |
| 17 | **Medium** | `vp9_scene_detection_onepass` "always called" under SVC? | **DEFERRED — Stage 2 LLD-svc** | A5 Q-RC-26 says "always called". Per A6 §3.7, each spatial layer calls `encode_frame_to_data_rate` separately, and scene detection is called inside `encode_without_recode_loop` per A1 §3.5 (`:4085`). So scene detection runs **per spatial layer, per superframe**. That's 3× compute for 3-layer spatial SVC. **A8 confirms the per-layer interpretation;** LLD-svc verifies and documents. |
| 18 | **Medium** | Frame dropping interactions across SVC layers | **DEFERRED — LLD-svc** | A6 owns SVC frame-drop bookkeeping (`svc.drop_spatial_layer[]`, `svc.last_layer_dropped[]`, `svc.framedrop_thresh[]`); A5 owns the drop decision (`vp9_rc_drop_frame`). Cross-layer drop semantics: if layer N drops, layer N+1 (which may reference it) is forced to drop. LLD-svc enumerates the exact cascade rules. |
| 19 | **High** | `is_altref_enabled` returns 0 for REALTIME+CBR — but VBR-RT? | **ACCEPTED — A1 patch (one-line clarification)** | A1 §3.5 already states "RT VBR can use altref if `lag_in_frames >= 4`". A1 §2.1 field doc for `enable_auto_arf` says "RT CBR effectively disables altref via `is_altref_enabled`". **Combined, the answer is: RT-VBR-with-lag-≥-4 DOES enable altref.** Stage 4a's MVP target (CBR) doesn't need it; Stage 4d (temporal SVC) might. **A1 patch:** add a one-line clarification in §3.5 invariants making this explicit. **No re-dispatch.** |
| 20 | **Low** | cpu-used 5 vs 6+ split is the `use_nonrd_pick_mode` boundary | **DEFERRED — PLAN.md scope clarification** | Captured implicitly in A6 §5. PLAN.md commits to cpu-used 6-8 priority. cpu-used 5 also uses non-RD path. **Recommendation:** test 6/7/8 in CI (per PLAN.md success criteria); allow cpu-used 5 as a "best quality real-time" knob with no CI gating. |
| 21 | **Low** | A6's "tile-rows always 0" assertion under SVC? | **DEFERRED — L-62** | Captured as L-62. Keep assertion in v1; revisit if SVC at 4K is added later. 4K SVC isn't in PLAN.md §2 targets. |
| 22 | **Low** | BufferPool 12-slot vs 8-ref-frame spec | **DEFERRED — D-08 / public API** | Captured as D-08. The 3 scaled slots are implementation detail and should not appear in the public API. |
| 23 | **Medium** | Sub-pixel filter selection in `search_filter_ref` not transcribed | **DEFERRED — LLD-mode-decision** | A2 §7 Q11 + L-48 already flag this. The non-RD path tries `EIGHTTAP` and `EIGHTTAP_SMOOTH` (not `EIGHTTAP_SHARP`); `force_smooth_filter` may drop `EIGHTTAP`. Stage 2 LLD reads `vp9_pickmode.c:search_filter_ref` to extract exact thresholds. |
| 24 | **Low** | Compound mode handling at speed 6+ claimed dead | **DEFERRED — L-46** | Captured as L-46. Drop compound mode from Stage 4 LLD; add divergence-ledger entry. A8 verifies: `comp_modes` is at most 2, only when `sf.use_compound_nonrd_pickmode` AND `usable_ref_frame == ALTREF_FRAME` (A2 §7 Q9). At cpu-used ≥ 5 default, `comp_modes = 0`. Compound path is **only reachable for VBR + lag + ≤720p** at speeds 5-6 (A6 §5.12 line 1560-1563). For RT-CBR-no-lag (the primary Stage 4a target), compound is dead. |
| 25 | **High** | External rate-control plug-in (EXT_RATECTRL) — WebRTC dependency | **DEFERRED — flag for Arthur (PLAN.md scope question)** | **PLAN.md scope question.** Already captured as R-06. PLAN.md doesn't mention EXT_RATECTRL but D3 commits to WebRTC viability. libwebrtc uses `cpi->ext_ratectrl` extensively for loss-aware RC. **Arthur must decide:** is EXT_RATECTRL in v1, v1.1, or v2? **A8 recommendation:** opt-in API surface in v1 (small surface area, ~5 call sites in libvpx). Flag this as a top-priority Arthur clarification. |
| 26 | **Medium** | `vp9_change_config` reallocation on dimension change | **DEFERRED — Stage 4a LLD** | LLD-encoder-shell documents the API contract for runtime config changes. Real-time use cases (bitrate ramp, layer drop) require graceful handling. |
| 27 | **Low** | Recode loop dead — but is the function dead, or just unused? | **DEFERRED — L-34** | Captured as L-34. Stage 2 LLD-arch confirms no speed-feature-dependent code in real-time touches recode-loop variables. **A8 verification:** A1 §3.4 (`:5645`) shows `sf.recode_loop == DISALLOW_RECODE` is the RT branch into `encode_without_recode_loop`; the alternate path (`encode_with_recode_loop`) is never entered. Variables like `cpi->frames_since_key` are also written from the RT path (RC bookkeeping), so they survive; only the `_with_recode_loop` function itself can be dropped. |
| 28 | **Low** | 4×4 transform paths in real-time — exercised? | **DEFERRED — LLD-transforms** | A2 §4 confirms non-RD path uses `TX_16X16` max (`tx_size_search_method = USE_TX_8X8` per A6 §5.12). But A8 notes: KF intra-only blocks **can hit TX_4X4** through `tx_mode = TX_MODE_SELECT` and the per-block tx-size decision in `select_tx_mode`. Stage 2 LLD-transforms confirms (likely keeps the kernel; the SIMD cost is small). |
| 29 | **High** | Speed-feature setter is per-frame condition-tuple dependent | **DEFERRED — R-10 / HLD-threading test harness** | Captured as R-10. Stage 2 HLD-threading documents the test-harness shape: condition-tuple replay, not just `(speed, w, h)`. |
| 30 | **High** | A4 §5 likely missing inter_mode probability updates | **ACCEPTED — same re-dispatch as #1** | This is the same A4 re-dispatch as challenge #1. Combined scope: verify `vp9_inter_mode_tree`, `default_inter_mode_probs`, `vp9_inter_mode_prob_diff` machinery are all transcribed. |
| 31 | **Low** | Trellis-quant skip on cpu-used 6-8 verified? | **DEFERRED — L-28** | Captured as L-28. A8 confirms: `sf->optimize_coefficients = 0` at cpu-used ≥ 5 (A6 §5.12 line 1514: `allow_skip_recode = 0` plus default `optimize_coefficients` left at 0 by RT cascade). Our Rust port can skip `vp9_optimize_b` entirely in v1. Divergence ledger. |
| 32 | **Medium** | VP9_COMP has ~120 fields; A1 lists ~30 explicitly | **REJECTED — A8 considers A1 sufficient** | A1 §2.4 reproduces VP9_COMP **verbatim** (the full 343-line `typedef struct VP9_COMP`), then provides field-by-field semantics grouped by subsystem (frame geometry, configuration mirror, per-frame transient, source/scaled buffers, TPL [non-RT], tile/partition state, reference-frame management, token buffers, RD/cost tables, MV cost LUTs, timestamps, rate control, cost tables filled per frame, speed features, two-pass, altref/denoiser/AQ, SVC, source-variance partitioning, per-SB partition cache, resizing, levels, multi-threading state, stats/misc, conditional/testing). The "~30 fields" count in A7 misreads the doc; **A1 covers all ~120 fields** by group. Stage 2 HLD-arch + LLD-encoder-shell expand the per-field type/ownership decisions, which is normal Stage 2 work. |

### 5.2 Detail on the high-severity A4 re-dispatch (challenges #1, #30)
<a name="challenge-blocker-detail"></a>

**Scope for re-dispatched A4:** verify that 04-bitstream-entropy.md §5
transcribes every load-bearing probability table verbatim. The minimum
list (sourced from libvpx file paths):

1. **Segmentation** (`vp9/common/vp9_seg_common.{h,c}`): segment-id
   tree, `vp9_default_pred_probs`, `vp9_default_segment_tree_probs`,
   `vp9_segment_tree`.
2. **Skip** (`vp9_entropymode.c`): default skip probs, skip tree.
3. **Tx-size** (`vp9_entropymode.c`): tx-size context probs, tx-size
   tree, default tx-size probs at TX_8X8 / TX_16X16 / TX_32X32.
4. **Intra modes — keyframe** (`vp9_entropymode.c`): `vp9_kf_y_mode_prob`
   (10 × 10 × `vp9_intra_mode_tree`), `vp9_kf_uv_mode_prob` (10 × `vp9_intra_mode_tree`).
5. **Intra modes — non-keyframe** (`vp9_entropymode.c`):
   `default_if_y_probs`, `default_if_uv_probs`, `default_intra_inter_prob`,
   `vp9_intra_mode_tree`.
6. **Inter modes** (`vp9_entropymode.c`): `default_inter_mode_probs`
   (per `INTER_MODE_CONTEXTS=7`), `vp9_inter_mode_tree`.
7. **Partition — keyframe** (`vp9_entropymode.c`): `vp9_kf_partition_probs`,
   `vp9_partition_tree`.
8. **Partition — non-keyframe** (`vp9_entropymode.c`):
   `default_partition_probs`, `vp9_partition_tree`.
9. **Interp filter** (`vp9_entropymode.c`):
   `default_switchable_interp_prob`, `vp9_switchable_interp_tree`.
10. **Compound / single reference** (`vp9_entropymode.c`):
    `default_single_ref_p`, `default_comp_ref_p`,
    `default_comp_inter_p`, `comp_ref_type` tree.
11. **MV** (`vp9_entropymv.{h,c}`): `default_nmv_context`,
    `vp9_mv_joint_tree`, `vp9_mv_class_tree`, `vp9_mv_class0_tree`,
    `vp9_mv_fp_tree`.
12. **Coefficients** (`vp9_entropy.{h,c}`): `default_coef_probs_4x4`,
    `default_coef_probs_8x8`, `default_coef_probs_16x16`,
    `default_coef_probs_32x32` (4D arrays:
    `[PLANE_TYPES][REF_TYPES][COEF_BANDS][COEFF_CONTEXTS][UNCONSTRAINED_NODES]`),
    `vp9_pareto8_full[255][PARETO8_NODES]`, `vp9_coef_tree`,
    `vp9_coef_con_tree`, `pt_energy_class[ENTROPY_TOKENS]`,
    `vp9_get_coef_band` translation table, every category extra-bit prob.

A4 must produce a checklist: ✓ present in §5 / ✗ missing. Missing items
get appended.

### 5.3 Disposition summary

- **ACCEPTED — re-dispatch:** A4 narrow (challenges #1, #30).
- **ACCEPTED — A2 patch:** Laplacian R-D tables (#2).
- **ACCEPTED — A1 one-line patch:** RT-VBR altref clarification (#19).
- **ACCEPTED — assigned to Stage 2 HLD-threading (no re-dispatch):** #7
  (worker-mutation catalogue, satisfied by §3 of this doc).
- **DEFERRED — Stage 2 LLDs:** #4, #5, #8-#15, #17, #18, #20-#24,
  #26-#28, #31.
- **DEFERRED — flag for Arthur (PLAN.md scope):** #3 (AQ modes 1/2),
  #25 (EXT_RATECTRL).
- **REJECTED:** #16 (loop count is upper bound, not array size); #32
  (A1 covers all ~120 fields by group).

Total: 4 ACCEPTED (1 re-dispatch + 2 patches + 1 catalogue task),
24 DEFERRED, 2 REJECTED, 2 flagged for Arthur. Severity-Blocker count:
1 (#2, addressed by A2 patch). Severity-High count: 10, all
dispositioned.

---

## 6. Verified Stage 0 assumptions

PLAN.md and Stage 0's plan made several implicit/explicit assumptions
about libvpx. Stage 1 either confirms or contradicts each.

| # | Assumption | Source | Verdict | Evidence |
|---|---|---|---|---|
| S0-1 | "Real-time path is largely separate from RDO" | PLAN.md §6 / Stage 4a "non-RD mode picker" | **CONFIRMED** | `sf.use_nonrd_pick_mode = 1` at cpu-used ≥ 5 (A6 §5.12); `sf.recode_loop = DISALLOW_RECODE` skips `encode_with_recode_loop` entirely (A1 §3.4 `:5645`). The non-RD path lives in `vp9_pickmode.c` (A2), entirely separate from `vp9_rdopt.c` (the RDO sibling, out of scope per A2 §1). The two paths share kernels (transforms, intra-pred, ME primitives, entropy) but not the mode-decision loop. |
| S0-2 | "SVC adds ~30% complexity on top of base encoder" | PLAN.md §6 Stage 4d/4e | **PARTIALLY CONFIRMED — likely an underestimate for spatial SVC** | A6 §1, §3.7-§3.8, §5 SVC clauses. Temporal SVC is small (~10-15% complexity: per-layer RC, layer-id bookkeeping, drop-frame rules). Spatial SVC is **much larger**: inter-layer prediction (A6 §3.8), `vp9_scale_references`, scaled-buffer pool slots, `simulcast_mode`, dual `ref_mode_set` (`ref_mode_set_svc` 8-entry vs `ref_mode_set` 12-entry, A2 §2.3), `nonrd_keyframe` SVC variants, per-layer scene detection (A7 challenge #17), separate `LAYER_CONTEXT.rc` (A5), `vp9_svc_constrain_inter_layer_pred`, `svc_use_lowres_part`, `force_zero_mode_spatial_ref`, `use_gf_temporal_ref_current_layer`. **More like 40-50% additional complexity for full spatial+temporal SVC.** PLAN.md Stage 4e (6-8 weeks) is consistent with this; the "30%" assumption was on the optimistic end. |
| S0-3 | "Threading mirrors a pthread tile-worker model" | PLAN.md D6 + Q3 | **CONFIRMED** | A6 §3 walks through pthread+mutex+condvar lifecycle in `vpx_thread.{h,c}`. `vp9_ethread.c` is one-worker-per-tile-column under tile-MT; row-MT adds a per-tile-column job queue with work-stealing. A6 §7.3 recommends `std::thread::scope` + hand-rolled `(Mutex<JobQueue>, Condvar)` as a 1:1 mapping. **The model is exactly pthread tile workers.** |
| S0-4 | "Denoiser is ~5k LOC" | PLAN.md §6 Stage 4c estimate | **CONFIRMED** | A5 §1: `vp9_denoiser.{h,c}` is 839 + 132 = ~970 LOC, plus `vp9_noise_estimate.{h,c}` 302 + 54 = ~360 LOC. Total **~1.3 KLOC of C** for denoiser + noise estimate. With SIMD kernels for SSE2 / NEON (denoiser hot loop), full SIMD-aware port lands at ~3-5 KLOC of Rust. **PLAN.md estimate is in the right zip code.** |
| S0-5 | "AQ is ~3k LOC, 3 modes" | PLAN.md §6 / §2 success criteria | **CONTRADICTED — three issues** | (a) **LOC**: A5 §1 source files: `vp9_aq_variance.{h,c}`, `vp9_aq_complexity.{h,c}`, `vp9_aq_cyclicrefresh.{h,c}`, `vp9_aq_360.{h,c}`. Aggregate roughly ~2.5 KLOC. Close to 3K. Confirmed. (b) **Mode count**: PLAN.md says "3 AQ modes matching libvpx". libvpx defines 8 AQ modes (NO_AQ=0, VARIANCE_AQ=1, COMPLEXITY_AQ=2, CYCLIC_REFRESH_AQ=3, EQUATOR360_AQ=4, PERCEPTUAL_AQ=5, PSNR_AQ=6, LOOKAHEAD_AQ=7). Per A1 §2.3 + A5 §3. So "3 modes" is correct as a target (modes 1/2/3), **but**: (c) **Reachability**: A5 §3 / A7 challenge #3 finds that **modes 1/2/4 are gated by `!CONFIG_REALTIME_ONLY`** (`vp9_encoder.c:4206-4239`). Only mode 3 (CYCLIC_REFRESH_AQ) is real-time-reachable in libvpx itself. **This is a PLAN.md scope mismatch** — D-05 in `_decisions-needed-draft.md` captures the choice. Arthur must decide whether to un-gate modes 1/2 in our port, drop them, or defer to v2. |
| S0-6 | "lookahead is push-pop FIFO" | PLAN.md implicit | **CONFIRMED** | A1 §2.9: `vp9_lookahead_{push,pop,peek}`. Capacity `MAX_LAG_BUFFERS=25`. Real-time CBR commonly runs with `lag_in_frames=0` or 1. |
| S0-7 | "Tile-threading max useful is 4-8 threads at 1080p" | PLAN.md §2 1080p30 / 1080p60 targets | **CONFIRMED** | A6 §1 + A1 §2.5: `MAX_NUM_THREADS=80`, `MAX_NUM_TILE_COLS=64`. But tile width is capped by `MIN_TILE_WIDTH_B64=4` (256 px) and `MAX_TILE_WIDTH_B64=64` (4096 px) (A6 §3.5). At 1080p (1920 px wide), max useful tile columns = 1920/256 = 7, log2 = 2 → max 4 tiles. At 1080p60, 4 tile workers is the natural cap; row-MT can use more, but adaptive_rd_thresh_row_mt determinism trade-offs apply (A6 §7.7). |
| S0-8 | "Frame-parallel encoding is in libvpx (Q4 implies it)" | PLAN.md Q4 implicit framing | **CONTRADICTED** | A6 §1 + A7 challenge #6: **libvpx has no encoder-side frame-parallel.** `cm->frame_parallel_decoding_mode` is a decoder hint written into the bitstream. The encoder is single-frame-in-flight. PLAN.md Q4 ("frame-parallel encoding in v1 vs v2") has no libvpx precedent to mirror. Adding frame-parallel encode is a v1+v2 design decision, not a port. **Recommendation:** answer Q4 as "deferred to v2" unless 1080p60 misses budget without it (D-04). |
| S0-9 | "Recode loop is non-RT only" | PLAN.md Stage 4a "single-pass" | **CONFIRMED** | A1 §3.5 invariant: `sf.recode_loop == DISALLOW_RECODE` set by real-time speed features. Single re-encode-on-overshoot branch in `encode_without_recode_loop` is the only conditional re-encode. |
| S0-10 | "Alt-ref is non-RT-CBR only" | PLAN.md implicit | **PARTIALLY CONFIRMED** | A1 §2.1 + A7 challenge #19: RT-CBR has altref disabled (`is_altref_enabled`). **RT-VBR with `lag_in_frames >= 4` and `enable_auto_arf` enables altref.** PLAN.md commits to both CBR and VBR; if Stage 4a's MVP is CBR-only (which it is — §6 Stage 4a focuses on CBR), altref can be deferred to a later stage that adds VBR-RT with lag. |
| S0-11 | "byte parity is a debugging tool, BD-rate is the gate" | PLAN.md D8 | **CONFIRMED** | All specialists treat byte parity as a diagnostic. A5 Q-RC-23 (denoiser SIMD), A6 §7.7 (`adaptive_rd_thresh_row_mt`), A7 challenges #4 and #8 are cases where libvpx itself doesn't guarantee byte parity. |
| S0-12 | "cpu-used 6-8 is the real-time range" | PLAN.md D1 | **CONFIRMED with caveat** | A6 §5: cpu-used 5 also uses non-RD path (`use_nonrd_pick_mode=1`); cpu-used 9 is below the real-time floor (experimental). The "primary" RT range is 6-8; cpu-used 5 can be supported as a "best quality real-time" knob with negligible additional implementation effort (A7 challenge #20). |
| S0-13 | "real-time path is what we port; non-RT code is excluded" | PLAN.md §8 | **CONFIRMED — large savings** | Excluding non-RT removes: 2-pass encoding (`vp9_firstpass.c`), encode_with_recode_loop, full RDO (`vp9_rdopt.c`), AQ modes 1/2/4/5/6/7, temporal filter (`vp9_temporal_filter.c`), TPL model, perceptual AQ k-means, lookahead-AQ, ARNR filter, mbgraph stats. Conservatively this is **40-50% of the encoder C source by line count**. Not a tight estimate; will firm up at HLD-arch. |

**Summary verdict:** Stage 0 assumptions were broadly accurate.
Three substantive corrections needed:

1. **S0-5 (AQ mode reachability)** — Arthur must decide D-05.
2. **S0-2 (SVC complexity)** — likely 40-50% additional, not 30%.
3. **S0-8 (encoder-side frame-parallel)** — does NOT exist in libvpx;
   Q4 has no precedent to mirror.

---

## 7. Stage 2 entry briefing

Single-page summary for the Stage 2 HLD authors (HLD-arch,
HLD-threading, HLD-simd, HLD-svc, HLD-bit-depth) and the ~15 LLD
authors.

### 7.1 What you (the Stage 2 author) must read before starting

- **Always:** PLAN.md (locked decisions D1-D9, deferred Q1-Q6).
- **Always:** This doc (00-overview.md) — §1 diagram, §2 timeline,
  §3 state graph.
- **Always:** `_decisions-needed-draft.md` to know which D-/L-/R- items
  your LLD must resolve.
- **HLD-specific:** the specialist sections cited in your scope below.

### 7.2 HLD authors — scope per HLD

**HLD-arch (00-architecture):**

- **Read:** A1 in full; A4 §2 (data structures shared with `rvp9-core`);
  A5 §2.1 (`RATE_CONTROL` placement); A6 §2 (`VP9_COMP` decomposition
  cross-cutting); §1, §2, §3, §4 of this doc.
- **Resolve:** D-01 (HBD scope), D-08 (buffer pool API), D-09
  (setjmp/longjmp Rust error model), L-29 (`VP9_COMP` decomposition),
  L-30 (lookahead bypass at lag-0), L-32 (pin to REALTIME mode), L-33
  (drop `CONFIG_*` knobs), L-34 (drop recode loop), L-37 (`Arc` vs
  `Rc` for frame pool), R-06 (EXT_RATECTRL scope — Arthur).
- **Stage-2 cross-cuts:** the FRAME_CONTEXT ring (§4.3 above), the
  error model decision, the public API surface.

**HLD-threading (01-threading):**

- **Read:** A1 §2.5 + §3.6 + §3.7; A6 in full (the threading specialist);
  §3, §4.1, §4.4, §4.9 of this doc.
- **Resolve:** D-03 (tile-threading primitives, recommendation: option 1
  from PLAN.md — `std::thread` + `Mutex`/`Condvar`/`Barrier`); D-10
  (SPEED_FEATURES representation, A6 §7.1 recommendation: snapshot into
  ThreadData); D-11 (`ThreadData::mb` aliasing, A6 §7.2 recommendation:
  scoped threads + slice splitting for tile-MT; UnsafeCell for row-MT);
  L-31 (eager vs lazy worker spawn); L-56 (`VPxWorker` → `std::thread::scope`
  mapping); L-57 (per-worker `FRAME_COUNTS` allocation); L-59
  (`row_mt` vs `max_threads` modeling); L-62 (assert `log2_tile_rows=0`
  in v1); L-63 (Windows thread priority); L-64 (golden-file
  speed-feature byte-parity test); R-08 (`adaptive_rd_thresh_row_mt`
  bit-exact-by-default + divergence ledger); R-09 (RTCD ordering);
  R-10 (condition-tuple replay for byte-parity testing).
- **Stage-2 cross-cuts:** the worker-mutation catalogue in §3.6 above is
  your starting checklist; the SPEED_FEATURES snapshot strategy
  determines how all other LLDs read it; the slice-splitting strategy
  determines the public surface of every shared buffer
  (`above_context`, `mi_grid_visible`, frame planes, segmentation map,
  denoiser running averages, etc.).

**HLD-simd (02-simd-dispatch):**

- **Read:** A1 §6 (RTCD init); A2 §6 (per-kernel SIMD inventory for
  non-RD); A3 §6 (ME SIMD: SAD, variance, subpel, integer projection,
  hadamard, satd); A4 §6 (boolean coder; mostly scalar); A5 §6
  (denoiser, AQ, scene detection SIMD); A6 §6 (threading-adjacent SIMD:
  `vp9_accumulate_frame_counts` is scalar-only; `vpx_lpf_*` for loop
  filter; `vpx_scaled_*` for SVC downscaling).
- **Resolve:** D-02 (RTCD dispatch table: hand-written Rust vs build.rs
  generator — recommend hand-written per `_decisions-needed-draft.md`
  rationale).
- **Stage-2 cross-cuts:** the BFP table (A1 §5) is the per-block-size
  variance/SAD dispatch wired up at encoder init; the `vpx_dsp_rtcd`
  table is the bulk of SIMD-able kernels and must complete before any
  worker spawns.

**HLD-svc (04-svc):**

- **Read:** A1 §3.4 + §3.9 (SVC frame-graph touch-points); A5 §2.1
  (`LAYER_CONTEXT.rc`); A6 §3.7-§3.8 (SVC start-layer + inter-layer
  pred); A6 §5.12-§5.18 (SVC-specific speed-feature deltas); §4.2 of
  this doc (cross-cutting).
- **Resolve:** L-36 (SVC reference-slot management); L-58 (SVC
  layer-context swap ordering between recode loops and frame loops);
  L-61 (SVC max-tile-cols sizing — pool sized to top-resolution layer);
  Plus the deferred A7 items: #17 (per-layer scene detection), #18
  (cross-layer drop semantics), #21 (tile-rows under SVC at 4K).
- **Stage-2 cross-cuts:** the layer iteration is owned in HLD-svc;
  per-layer rate control is in LLD-rate-control; per-layer denoiser
  bookkeeping is in LLD-denoiser.

**HLD-bit-depth (03-bit-depth):**

- **Read:** A1 §2.1 (HBD config field); A4 §2 (`tran_low_t` HBD
  conditional); A5 §2 (HBD-specific quant); §3.1 of this doc and A7
  challenge #11.
- **Resolve:** D-01 / Q1 (HBD scope).

### 7.3 LLD authors — scope per LLD

| LLD | Read first | Primary source | Stage-2 keys to resolve |
|---|---|---|---|
| `bitstream.md` | A4 §1-§4 | `vp9_bitstream.c` | L-11, L-13, L-14, L-15 |
| `entropy.md` | A4 §2, §5 (re-dispatched first) | `vp9_entropy*.c`, `vp9_subexp.c` | L-09, L-10, L-27 |
| `transforms.md` | A4 §6 (scan orders); A7 #28 (4×4 reachability) | `vpx_dsp/{inv,fwd}_txfm*.c` | (Stage 3 LLD) |
| `quant.md` | A4 §2 (quant tables); A5 §2 (quant in RC) | `vp9_quantize.{h,c}` | (Stage 3 LLD) |
| `intra-pred.md` | A2 §4 (4-mode intra in RT); A4 §6 | `vpx_dsp/intrapred*.c` | (Stage 3 LLD) |
| `loopfilter.md` | A1 §3.4 step 47; A6 §6 (LF threading) | `vp9_loopfilter.c`, `vp9_thread_common.c` | (Stage 3 LLD) |
| `frame-buffers.md` | A1 §2.8 (BufferPool); §3.9 (rotation) | `vp9_onyxc_int.h` BufferPool | D-08, L-37 |
| `motion-estimation.md` | A3 in full | `vp9_mcomp.c` | L-05, L-06, L-07, L-08 |
| `mode-decision.md` | A2 in full + A7 #2 (Laplacian R-D, blocker) | `vp9_pickmode.c`, `vp9_rd.c` | L-38, L-39, L-40, L-41, L-42, L-43, L-44, L-45, L-46, L-47, L-48, L-49, L-50, L-51, L-52, L-53, L-54, L-55 |
| `partition.md` | A2 §3 (`choose_partitioning`) | `vp9_encodeframe.c` | L-39, L-52, L-53, L-54 |
| `rate-control.md` | A5 §1-§4 | `vp9_ratectrl.c` | L-01, L-04, L-16, L-17, L-18, L-19, L-23, L-25, L-26, R-04, R-05 |
| `denoiser.md` | A5 (denoiser sections) | `vp9_denoiser.c` | L-22, Q-RC-8/9/10/11; R-03 |
| `aq.md` | A5 §3 + D-05 decision (Arthur) | `vp9_aq_*.c` | L-02, L-03, L-21, L-24, D-05, D-06 (ROI) |
| `svc.md` | A6 §3.7-§3.8; §4.2 of this doc | `vp9_svc_layercontext.c` | L-36, L-58, L-61, L-62, #17, #18 |
| `tile-threading.md` | A6 §3 + §3 of this doc | `vp9_ethread.c`, `vp9_multi_thread.c` | D-11, L-31, L-56, L-57, L-59 |

### 7.4 Critical-path items for Arthur to clarify before Stage 2 starts

1. **D-01 (HBD in v1?)** — PLAN.md Q1. Recommend: no.
2. **D-05 (AQ modes 1/2 in v1?)** — PLAN.md §2 mismatch. Recommend:
   ship mode 3 only; edit PLAN.md.
3. **R-06 (EXT_RATECTRL scope?)** — WebRTC dependency. Recommend:
   opt-in API surface in v1.

### 7.5 Items the synthesizer flags as still-open even after disposing of A7

These were already in `_decisions-needed-draft.md`, but A8 considers
them load-bearing enough to surface in this briefing:

- **L-19 (`vp9_encodedframe_overshoot`)** — single most opaque
  mutation in the encoder. Stage 4a needs explicit byte-parity testing
  under scene-change inputs.
- **L-22 (denoiser SIMD bit-parity)** — strategy for differential
  testing when libvpx itself isn't bit-exact.
- **R-08 (`adaptive_rd_thresh_row_mt`)** — bit-exact-by-default
  divergence at cpu-used 6.
- **R-10 (speed-feature condition tuples)** — byte-parity CI needs
  condition-tuple replay, not (speed, w, h).
- **L-19 + the missing Laplacian R-D tables** (A7 #2) — together,
  these form the highest-risk Stage 4a items.

### 7.6 What Stage 2 does NOT need to read (specialists' out-of-scope)

- 2-pass encoder (`vp9_firstpass.c`, `vp9_temporal_filter.c`, TPL
  model, ARNR filter, mbgraph).
- AQ modes 5/6/7 (perceptual, PSNR, lookahead).
- `encode_with_recode_loop` (non-RT).
- `vp9_rdopt.c` (full RDO).
- `CONFIG_*` testing knobs (`CONFIG_RATE_CTRL`, `CONFIG_NON_GREEDY_MV`,
  `CONFIG_COLLECT_COMPONENT_TIMING`, `CONFIG_INTERNAL_STATS`,
  `CONFIG_BETTER_HW_COMPATIBILITY`).

---

## Cross-references

| Specialist | File | Sections heavily cited above |
|---|---|---|
| A1 (architecture) | `01-architecture.md` | §1, §2, §3.4-§3.7, §3.9, §7 |
| A2 (encoder core) | `02-encoder-core.md` | §1, §2, §3-§4, §6, §7 |
| A3 (motion estimation) | `03-motion-estimation.md` | §1, §2 |
| A4 (bitstream + entropy) | `04-bitstream-entropy.md` | §1, §2, §3 (rest pending re-dispatch) |
| A5 (rate control + denoiser + AQ) | `05-rate-control.md` | §1, §2.1, §3, §6, §7 (30 Q-RC items) |
| A6 (threading + SVC + speed features) | `06-threading-svc.md` | §1, §2, §3, §5.12-§5.19, §6, §7 |
| A7 (devil's advocate) | `07-devils-advocate.md` (= `07-devils-advocate-driver.md`) | All 32 challenges |
| draft decisions | `_decisions-needed-draft.md` | D-01..D-11, L-01..L-64, R-01..R-10 |

---

*Document version: 0.2.0 — A8 synthesizer (supersedes manual fallback v0.1.x).*
