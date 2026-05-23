# HLD — Q.1.C Frame-Parallel VP9 Decode

**Author:** Q.1.C design agent
**Date:** 2026-05-23
**Status:** SIGNED OFF by lead 2026-05-23 (scoped to P0+P1 for this /goal; P2-P4 conditional on Q.1.A measurement)

## 1. One-line summary

Introduce a frame-pipeline driver above the current per-frame `Decoder` that admits up to `N` superframes/frames in flight, each running on a dedicated decoder instance whose only mutable cross-frame coupling is the post-decode ref-map / FRAME_CONTEXT publication; existing tile parallelism stays inside each frame worker.

## 2. Reference architecture (libvpx)

**Surprise finding (flagged for Arthur).** The libvpx checkout at `third_party/libvpx/` is the **stripped, tile-only** variant. The frame-parallel `frame_workers` array that lives in upstream libvpx's `vpx_codec_alg_priv` is **absent here**: `vpx_codec_alg_priv` carries a single `VP9Decoder *pbi` (`third_party/libvpx/vp9/vp9_dx_iface.h:22`), and `VP9Decoder` itself holds only `tile_workers` and `lf_worker` (`third_party/libvpx/vp9/decoder/vp9_decoder.h:107-108`). The cross-frame sync primitive `VP9FrameWorker` is not in this tree either — `vp9_thread_common.{c,h}` only exposes `VP9LfSync` for within-frame loop-filter rows (`third_party/libvpx/vp9/common/vp9_thread_common.c:38-58`). The `frame_parallel_decoding_mode` *flag* is still parsed (`vp9_decodeframe.c:2783`) and still gates adaptation (`vp9_decodeframe.c:3047`, also `:1978, :2069, :2375, :2388, :2432, :2476, :2515, :2560`), and `vp9_decoder.c:484` still carries the comment "Update progress in frame parallel decode" — fossils of the removed mechanism.

What this means for us: we do **not** have an in-repo libvpx oracle to mirror line-for-line. We reconstruct the canonical libvpx-upstream frame-parallel architecture (publicly documented):

- A pool of `N` `VP9Decoder` instances, sized at codec init via `cfg.threads` (`vpxdec.c:629-630`).
- A shared `BufferPool` of `RefCntBuffer` holding all decoded frames; `cm->ref_frame_map[8]` is per-decoder and stores `int` indices into this pool — exactly the indirection we already mirror at `crates/rvp9-decoder/src/lib.rs:266-269,311-326`.
- The demuxer round-robins frames into worker slots; each worker independently decodes its frame and, when done, atomically publishes the resulting `ref_frame_map` rotation and (if `frame_parallel_decoding_mode == 0`) a new `FRAME_CONTEXT` bank.
- Output ordering is by submission order.
- On streams with `frame_parallel_decoding_mode == 0`, backward adaptation runs (`vp9_decodeframe.c:3047-3054`), so frame `t+1` depends on frame `t`'s adapted context: frames must drain serially.

## 3. Proposed rvp9 design

### 3.1 Components

- **`FramePipeline`** — new struct in `rvp9-decoder` that owns `Vec<DecoderInstance>` (size = `num_frame_workers`) plus a `WorkerPool` of N frame threads. Front of the pipeline = `parse_packet`; back = an output FIFO of `(submission_index, FrameBuf, show_frame)`.
- **`DecoderInstance`** — a restructuring of today's `Decoder` (`crates/rvp9-decoder/src/lib.rs:259-560`) carrying everything per-stream: `frame_contexts[4]`, `coef_counts`, `mv_counts`, `mode_counts`, `lf_deltas`, `mv_grids[8]`, `prev_frame_mv_grid`, `prev_segment_ids`, `seg_feature_data`, `established_chroma`, `last_frame_*`. One **canonical** instance (stream state) and **N transient clones** active concurrently, each pre-loaded at admission.
- **`SharedFramePool`** — replaces `DecoderFramePool` (`crates/rvp9-core/src/frame.rs:1230`). Same 9-slot Arc storage, but `Arc<Mutex<SharedFramePool>>` so workers can `acquire_free`/`assign_slot` concurrently. Hot reads stay lock-free.
- **`StreamProgress`** — small `Mutex<StreamState>` holding the publishing side: `ref_frame_map[8]`, `ref_frame_sizes`, `prev_frame_mv_grid`, `prev_segment_ids`, `seg_feature_data`, `established_chroma`, plus the four `FRAME_CONTEXT` banks.

### 3.2 Reference-frame map ownership and locking

**Key insight:** because `FrameBuf = Arc<Yv12Buffer>` (`crates/rvp9-core/src/frame.rs:1154`), and a frame is *only ever read by other frames after it is fully decoded* (VP9 has no row-level cross-frame dependency at decode), we do **not** need libvpx-upstream's per-row `VP9FrameWorker` sync primitive. Each `DecoderInstance` is handed a **snapshot** of `ref_frame_map[8]` at admission (eight `Arc<Yv12Buffer>` clones); during decode it never touches the shared map. After decode, the pipeline acquires `StreamProgress` and applies the `refresh_frame_flags` rotation.

This is a deliberate weakening of libvpx's row-sync model. Costs nothing on quality (output bit-exact) and pays for itself in mutex / unsafe budget.

### 3.3 Sequencing — when can a frame start?

A frame can be admitted to a free worker iff:

1. Its `ref_frame_map` predecessors are all *committed* to `StreamProgress` (auto if frames commit in submission order).
2. Its FRAME_CONTEXT bank is what we believe will be current after all prior frames commit. On `frame_parallel_decoding_mode == 1` streams this is trivially the bank at admit-time. On `frame_parallel_decoding_mode == 0` streams — frame `t+1` reads bank-adapted-by-frame-`t` — admission waits for `t` to commit.

### 3.4 Output ordering

`parse_packet` already returns up to 8 `ParsedFrame`s per superframe in stream order (`crates/rvp9-decoder/src/lib.rs:848-921`). The pipeline tags each with a monotonic `submission_index` at admit; the output FIFO is a `BinaryHeap` keyed on that index so consumers see frames in source order regardless of completion order.

### 3.5 Coexistence with tile parallelism (Q.1.B)

Outer pool = frame workers (size `F`), inner = tile workers per `DecoderInstance` (size `T`). Total = `F × T` — proposed default `F = min(host_cores / max(num_tiles, 1), 4)` and `T = max(host_cores / F, 1)`. `Decoder::with_threads(N)` (`crates/rvp9-decoder/src/lib.rs:639`) augmented with `Decoder::with_frame_threads(F).with_tile_threads(T)`; today's `with_threads(N)` becomes `F = 1, T = N` (preserves Q.1.B behavior).

## 4. API impact

**Recommendation:** keep `decode_frame` as the single user-facing entry point; transparently pipelined when `F > 1`. Rationale:

- `parse_packet` already returns multiple `ParsedFrame`s per superframe (`crates/rvp9-decoder/src/lib.rs:848-921`).
- The harness uses `for frame in parsed { decoder.decode_frame(&frame) }` (`crates/rvp9-harness/tests/frame_parallel_regression.rs:96-105`).
- Bit-exactness preserved by output FIFO.

**Cost:** `decode_frame` may return a frame submitted several calls ago; we add a `Decoder::flush()` to drain at end-of-stream. The `WrongFrameKind` / `EmptyInterRef` / `EmptySlot` error path becomes asynchronous — document this in the type-level invariant comment block at `crates/rvp9-decoder/src/lib.rs:159-187`.

Existing single-threaded callers (`F = 1`, default) see zero behavioural change.

## 5. Phasing (LEAD-SCOPED)

This /goal lands **P0 + P1 only**. P2-P4 are deferred pending Q.1.A measurement results.

| Phase | Scope | Exit gate | This /goal? |
|---|---|---|---|
| **P0 — Infra factoring** | Split `Decoder` into per-stream `StreamProgress` and per-frame `DecoderInstance`; `Arc<Mutex<SharedFramePool>>`; no behavioural change, no concurrency. | 201/0/0/0/201 conformance + `frame_parallel_regression` + `profile1_decode` all green at `F = T = 1`. | **YES** |
| **P1 — Two-frame pipeline, `frame_parallel = 1` only** | Land `FramePipeline` with `F = 2`. Flag-0 streams fall back to serial admission. Tile pool stays at `T = 1`. | All P0 gates plus new harness test `frame_parallel_pipeline_two_workers` (`foreman-fp1.ivf` at `F = 2`, identical visible MD5s to existing `F = 1` ledger). | **YES** |
| **P2 — Widen frame pool** | `F = host_cores / T` with auto-detect; cap 8. Add `Decoder::with_frame_threads`. | Same gates as P1 + throughput ≥ `min(F, frame_count) × 0.7` of `F = 1`. | NO (deferred) |
| **P3 — Pipeline × tiles** | Allow `F > 1 && T > 1` simultaneously. | All gates + stress test `F = 2, T = 4` bit-exact. | NO (deferred) |
| **P4 — Bench + tune + policy** | Real-world throughput sweep; default-on policy. | Decision recorded. | NO (deferred) |

P0 + P1 are the load-bearing wave. The lead's rationale for scoping out P2-P4: we have **no measured perf gap** vs PLAN gates today; Q.1.A delivers that measurement. P2-P4 will be re-justified (or shelved) on the basis of that data.

## 6. Bit-exact gating

`read_triangulated_ledger` (`crates/rvp9-harness/src/md5file.rs`, consumed at `crates/rvp9-harness/tests/frame_parallel_regression.rs:85` and `tests/profile1_decode.rs:41`) is the oracle:

- **Triangulation** (vpxdec MD5 ∧ ffmpeg MD5 agree before trusting either) — `frame_parallel_regression.rs:81-85`.
- **Per-shown-frame** granularity catches misordered output.
- **No-skip enforcement** — `frame_parallel_regression.rs:73-77, 142-149`.

At each phase: (a) 201-vector official corpus (all `frame_parallel = 0`); (b) `frame_parallel_regression` (`frame_parallel = 1` clip); (c) `profile1_decode` (cross-chroma).

**Missing coverage to add at P1:** existing fixture is 30 frames. Add at least one longer `frame_parallel = 1` clip (≥ 120 frames, with show_existing_frame and altref) so the reorder buffer is genuinely exercised. MD5 ledger regen via `scripts/encode-corpus.ps1`.

## 7. Performance expectation

Theoretical ceiling = `min(F, average_in_flight_frames)`. Expected real-world `F = 4, T = 1` on a 4-core host: **~2.5–3.0×** on a typical `frame_parallel = 1` clip. Flag-0 streams: **zero speedup** (serialised by §3.3 condition 2).

**Policy for flag-0 streams:** stay serial. Speculation on FRAME_CONTEXT explodes unsafe / locking budget. P4 decision.

## 8. Risks & alternatives

1. **Do nothing.** Q.1.B tile parallelism + SIMD may already meet PLAN gates. If real-world clips average ≥ 4 tiles, tile parallelism alone may saturate cores. **Mitigation:** P0 ships zero-behaviour-change refactoring valuable even if P1+ deferred (the `StreamProgress` / `DecoderInstance` split is good hygiene). **The lead has accepted this risk and scoped P2-P4 out.**
2. **Single-tile clips dominate (common at low bitrate).** Tile parallelism gives nothing; frame parallelism is the only lever. Argues *for* P1.
3. **Hidden cross-frame state.** Inventory (`crates/rvp9-decoder/src/lib.rs:259-560`): `frame_contexts[4]`, `coef_counts`, `mv_counts`, `mode_counts`, `lf_deltas`, `have_resync`, `established_chroma` (`:401`), `mv_grids[8]` (`:416`), `prev_frame_mv_grid` (`:428`), `prev_frame_mv_grid_dims`, `last_frame_size`, `last_intra_only`, `last_show_frame`, `last_frame_was_key`, `prev_segment_ids` (`:462`), `prev_segment_ids_dims`, `seg_feature_data` (`:480`), `seg_abs_delta`, `last_frame_mi_grid` (`:503`), `last_frame_mi_dims`, `ref_frame_sizes`, `ref_frame_map`. Every one is a candidate for racing under pipelining; P0's job is to corral all into `StreamProgress`. The `established_chroma` overlay in `parse_packet` (`:895-903`) is a *parse-time* mutation — flag in P0 review.
4. **`mem::forget` soundness on the new outer pool.** Reuse the existing tile pool's closure-scope pattern (`crates/rvp9-core/src/tile_threading/mod.rs:20-29`). Do not invent a new soundness model.
5. **Speculative execution on flag-0 streams.** Rejected — §7 policy.

## 9. Open questions

1. **"vpxenc approach" wording.** Reading strictly, frame-parallel decode lives in libvpx's *decoder*. Encoder only sets the *flag*. Proceeding on assumption "vpxenc" is a slip for "vpxdec." **Lead has confirmed this interpretation in the kickoff journal.**
2. **v1 vs v2.** The agent's recommendation was v2. **Lead has scoped this /goal to P0+P1 (limited v1 commitment); P2-P4 will be re-decided post-Q.1.A.**
3. **Track F interaction.** Encoder (Track F) doesn't currently share decoder thread infra. If Track F lands an encoder-side `WorkerPool` reuse of `crates/rvp9-core/src/tile_threading/pool.rs`, the frame-pool factoring proposed here generalises cleanly. **Confirm with Track F before P0 lands.**
4. **Heterogeneous-tile streams.** P3 detail-design item, not P0 blocker.

## 10. Lead sign-off conditions

The HLD is signed off subject to:

1. **Scope is P0+P1 only for this /goal.** P2-P4 require fresh justification post Q.1.A measurement.
2. **Q.1.A must complete first.** Its numbers inform whether P1 alone is sufficient or whether we revisit P2-P4 immediately.
3. **P0 must be a true no-op refactor.** Bit-exact gates green on commit. Any deviation = back to design.
4. **The `established_chroma` parse-time mutation gets explicit treatment in the P0 PR description.**

### Critical Files for Implementation

- `D:\language\rvp9-2\crates\rvp9-decoder\src\lib.rs` (split `Decoder` → `FramePipeline` / `DecoderInstance` / `StreamProgress` in P0)
- `D:\language\rvp9-2\crates\rvp9-core\src\frame.rs` (wrap `DecoderFramePool` with `Mutex`, rename to `SharedFramePool` in P0)
- `D:\language\rvp9-2\crates\rvp9-core\src\tile_threading\pool.rs` (reuse `WorkerPool` pattern for frame-level pool in P1)
- `D:\language\rvp9-2\crates\rvp9-harness\tests\frame_parallel_regression.rs` (load-bearing flag-1 oracle; add `F = 2` and longer-clip variants in P1)
- `D:\language\rvp9-2\crates\rvp9-harness\tests\profile1_decode.rs` (orthogonal axis oracle)
