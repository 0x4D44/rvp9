# HLD-arch — top-level Rust architecture for rvp9-2

**Status:** draft v0.1.0, Stage 2 load-bearing HLD.
**Author:** HLD-arch (Stage 2).
**Date:** 2026-05-16.
**Inputs read (in order):** `PLAN.md`; `notes/vp9-analysis/00-overview.md`;
`01-architecture.md`; `decisions-needed.md` (Stage 2 entry — Arthur scope
RESOLVED block at the top is authoritative); skim of
`02-encoder-core.md`, `03-motion-estimation.md`, `04-bitstream-entropy.md`,
`05-rate-control.md`, `06-threading-svc.md`.

**Authoritative scope deltas from Stage 2 entry (`decisions-needed.md` §"Stage 2 entry — Arthur scope"):**

| Item | Scope |
|---|---|
| **D-01 (HBD)** | 8-bit only in v1. HBD deferred to Stage 6+. |
| **D-05 (AQ)** | CYCLIC_REFRESH (mode 3) only. |
| **D-13 (altref)** | CBR + VBR-RT, NO altref in v1. |
| **R-06 (EXT_RATECTRL)** | In v1. |

This document is the load-bearing top-level Rust HLD. Every other Stage 2
HLD (`01-threading.md`, `02-simd-dispatch.md`, `04-svc.md`) and every
Stage 2 LLD anchors to the crate boundaries, module graph, type system,
error model, `unsafe` policy, integer-width policy, pixel-type policy,
EXT_RATECTRL trait surface, build-time feature matrix, and rustdoc
policy defined here.

**Note on out-of-scope HLDs.** Per the Stage 2 entry scope, `03-bit-depth.md`
is no longer needed (D-01 RESOLVED); the pixel-type policy is owned by
this document (§7).

---

## 0. Document map

- [§1 Crate boundaries](#1-crate-boundaries)
- [§2 Module graph within `rvp9-core`](#2-module-graph-within-rvp9-core)
- [§3 Type-system overview](#3-type-system-overview)
- [§4 Error model](#4-error-model)
- [§5 `unsafe` policy (load-bearing)](#5-unsafe-policy-load-bearing)
- [§6 Integer-width policy](#6-integer-width-policy)
- [§7 8-bit-only pixel-type policy (load-bearing)](#7-8-bit-only-pixel-type-policy-load-bearing)
- [§8 EXT_RATECTRL plug-in interface](#8-ext_ratectrl-plug-in-interface)
- [§9 Build-time configuration](#9-build-time-configuration)
- [§10 Documentation policy](#10-documentation-policy)
- [§11 Open questions](#11-open-questions)

---

## 1. Crate boundaries

### 1.1 The existing six-crate split

PLAN.md §5 plus the Stage 0 completion log fix the workspace as
**six crates** under `crates/`:

```
crates/
├── rvp9-core/      shared primitives (bitstream, entropy, transforms,
│                   intra-pred, loop-filter, quant, frame mgmt, simd)
├── rvp9-decoder/   spec-conformant VP9 decoder, depends on -core
├── rvp9-encoder/   real-time encoder, depends on -core
├── rvp9-cli/       rvp9enc / rvp9dec binaries
├── rvp9-bench/     realtime + BD-rate benchmark harnesses
└── rvp9-harness/   conformance + differential test runners
```

Per `Cargo.toml`: this is a virtual workspace, `resolver = "2"`,
`edition = "2024"`, MSRV `1.95`. Lints `unsafe_op_in_unsafe_fn = warn`
and `missing_docs = warn` apply workspace-wide.

### 1.2 Review against Stage 1 findings — is the split right?

Yes, with one clarification recorded below. The Stage 1 synthesis
(00-overview §1, §3.6, §7) confirms three architectural facts that
keep the existing split correct:

1. **`rvp9-core` is the right home for everything a decoder needs.**
   The decoder is spec-defined and only requires: bitstream reader,
   range coder, probability tables (FRAME_CONTEXT, default probs,
   coefficient model — 00-overview §4.3), transforms (Stage 1 A4 §2,
   §6), intra prediction (4 RT modes per A2 §4 plus the full 10 modes
   the decoder requires), loop filter (A1 §3.7), quant/dequant tables
   (A4 §2), and frame buffer management (A1 §2.8). All of these are
   also consumed by the encoder.
2. **The encoder adds substantial state that no decoder needs.**
   `VP9_COMP` (A1 §2.4, ~120 fields), RATE_CONTROL (A5 §2.1, ~70
   fields), SPEED_FEATURES (A6 §2.11, 600+ lines), partition/mode
   decision (`vp9_pickmode.c`, A2), motion estimation (`vp9_mcomp.c`,
   A3), denoiser, AQ, SVC layer state — all encoder-private. They
   belong in `rvp9-encoder`, not `rvp9-core`.
3. **The decoder is the oracle.** Stage 3 ships the decoder first and
   validates every shared kernel under conformance vectors before the
   encoder is wired up. Locating shared kernels in `rvp9-core` lets the
   decoder exercise and validate them in isolation (PLAN.md Stage 3
   §1–§8 task list).

The clarification: **`rvp9-core` must also own the EXT_RATECTRL trait
definition** (§8 below) even though only the encoder ever invokes it.
The trait lives in `rvp9-core::ratectrl_ext` so that downstream WebRTC
crates can depend on `rvp9-core` (small, stable surface) without
pulling in the encoder. This costs nothing — the trait has no data,
just method signatures.

### 1.3 No new crates

Considered and rejected:

- **`rvp9-simd` separate crate** — splitting kernels from their
  dispatch tables is a layering violation; SIMD lives as
  `rvp9-core::simd`.
- **`rvp9-tables` separate crate** — probability tables, quant
  tables, scan orders, and `vp9_prob_cost[256]` (D-14) are
  tightly coupled to consumers; co-locate.
- **`rvp9-svc` separate crate** — SVC is encoder-only (decoder only
  reads the superframe index per A4 §3); lives as `rvp9-encoder::svc`.
- **`rvp9-ffi` C-ABI crate** — out of scope for v1 (PLAN.md §8).

### 1.4 Per-crate visibility rules

| Crate | Public API surface | Internal-only |
|---|---|---|
| `rvp9-core` | bitstream reader/writer; entropy decoder/encoder; FRAME_CONTEXT; FRAME_COUNTS; transforms (forward + inverse); quant/dequant; intra predictors; loop filter; frame buffer pool; SIMD dispatch tables; `RateControllerExt` trait (§8); `Pixel = u8` aliases (§7); error types | per-arch intrinsic module bodies (`#[doc(hidden)]` even if `pub`) |
| `rvp9-decoder` | `Decoder` type, `decode_frame()` / `decode_superframe()`, configuration | tile workers, frame pool acquire/release wiring |
| `rvp9-encoder` | `Encoder` type, `encode_frame()`, `RateControllerExt` impl receipt, `EncoderConfig`, SVC layer config, AQ config (mode 3 toggle), denoiser toggle | rate control internals, partition/mode/ME, speed-feature cascade, tile workers, denoiser state |
| `rvp9-cli` | binaries `rvp9enc`, `rvp9dec`; no library API |  |
| `rvp9-bench` | benchmark binaries / library entry points |  |
| `rvp9-harness` | conformance + differential test runners |  |

`#[doc(hidden)]` is applied to items that are `pub` for cross-module
reasons (workspace internal) but not part of the API contract. Crate
authors must mark them explicitly; the lint `missing_docs = warn` will
otherwise force documentation on every `pub` item.

### 1.5 Cargo features and their crate scope

See §9 for the full feature matrix. Important for crate boundaries:

- `rvp9-core` exposes the lowest-level feature flags (`simd-avx2`,
  `simd-sse41`, `simd-sse2`, `simd-neon`, plus scalar-only) consumed
  by both decoder and encoder.
- `rvp9-encoder` exposes high-level feature flags (`denoiser`, `svc`,
  `ext-ratectrl`) which the user enables. The decoder has no such
  optional features in v1; it must always handle every conforming
  bitstream.

---

## 2. Module graph within `rvp9-core`

The shared crate is decomposed by **subsystem**, not by "what the
encoder calls into vs. what the decoder calls into". The decoder reads
the same probability tables, runs the same inverse transforms, computes
the same intra predictions, and runs the same loop filter as the
encoder's reconstruction step (A1 §3.7).

### 2.1 Top-level module list

```
rvp9-core/
└── src/
    ├── lib.rs              re-exports, error types, Pixel alias (§7)
    ├── bitstream/          byte reader/writer + range coder (vp9_bool*.c)
    ├── entropy/            FRAME_CONTEXT, FRAME_COUNTS, default tables,
    │                       backward adaptation, vp9_prob_cost[256] (D-14)
    ├── transforms/         forward + inverse DCT/ADST (4/8/16/32), scan
    │                       orders, IDCT-32 special cases
    ├── quant/              quantizer tables, quant/dequant kernels
    ├── intra_pred/         10 intra modes × {4,8,16,32} block sizes
    ├── loopfilter/         deblocking kernel + filter-level tables
    ├── frame/              YV12 frame buffer, FramePool, RefCntBuffer
    ├── color/              color-space + range enums (8-bit only)
    ├── geom/               BlockSize, TxSize, PlaneType, MotionVector,
    │                       MAX_MB_PLANE = 3
    ├── simd/               per-arch dispatch + scalar fallback
    │   ├── dispatch.rs     RTCD-equivalent (HLD-simd owns design)
    │   ├── x86_64.rs       SSE2/SSE4.1/AVX2 intrinsics
    │   ├── aarch64.rs      NEON intrinsics
    │   └── scalar.rs       reference scalar implementations
    └── ratectrl_ext/       RateControllerExt trait (§8)
```

Stage 1 supports each module's existence:

| Module | Stage 1 anchor |
|---|---|
| `bitstream` | A4 §1, §3 (`vp9_pack_bitstream`, range coder); 00-overview §4.3, §7.3 LLD-bitstream |
| `entropy` | A4 §2, §5 (verbatim tables); 00-overview §4.3, §4.4; D-14 (`vp9_prob_cost[256]`) |
| `transforms` | A4 §6 (scan orders); A2 §4 (TX_4X4..TX_16X16 in RT path); A7 #28 |
| `quant` | A4 §2 (quant tables); A5 §2 (quant in RC); A1 step 25 `vp9_set_quantizer` |
| `intra_pred` | A2 §4 (4-mode intra in RT — DC/V/H/TM); decoder needs all 10 modes |
| `loopfilter` | A1 §3.7 (`loopfilter_frame`); A6 §2.7 (`VP9LfSync`); Stage 1 follow-up flagged need for re-dispatch when LLD-loopfilter starts |
| `frame` | A1 §2.8 (12-slot pool, 8 ref + 1 cur + 3 scaled); D-08, L-37 |
| `simd` | A1 §6 (RTCD init); HLD-simd owns design; Q2 / D-02 |
| `ratectrl_ext` | R-06 RESOLVED; A1 §7.7 |

### 2.2 Dependency direction within `rvp9-core`

Strict, acyclic, leaf-towards-root:

```
geom ─┬─► (leaf)
color ─┘
            ┌─► quant ────┐
            │             │
transforms ─┤             ├─► entropy
            │             │
intra_pred ─┘             │
                          │
bitstream ────────────────┘
                          │
frame ────────────────────┤
                          │
loopfilter ───────────────┘

simd: depended on by transforms, intra_pred, loopfilter, quant.
ratectrl_ext: depended on by no -core module; pulled in by rvp9-encoder.
```

Two rules to keep this clean:

- `geom` and `color` are leaves and have no internal dependencies. They
  contain `BlockSize`, `TxSize`, `MotionVector`, `PlaneType`,
  `BitstreamProfile` (0 only, in v1), and enums for color space /
  range. Pure data, no kernels.
- The `simd` module is **depended on**, never depends. The kernel
  modules (transforms, intra_pred, loopfilter, quant) call into
  `simd::dispatch` to pick the right implementation. The dispatch
  tables themselves contain function pointers that ultimately resolve
  to per-arch intrinsic functions in `simd::x86_64` / `simd::aarch64`
  or scalar fallbacks in `simd::scalar`. **HLD-simd** owns the precise
  shape of this dispatch (Q2 / D-02). HLD-arch fixes only that the
  module exists in `rvp9-core` and is initialized once-globally before
  any worker spawns (A1 §6, R-09, 00-overview §4.9).

### 2.3 `rvp9-encoder` module graph (informative)

Outside the load-bearing scope of HLD-arch, but stated here so the
crate boundary in §1 is concrete. Detailed module decisions belong to
the individual encoder LLDs.

```
rvp9-encoder/
└── src/
    ├── lib.rs              Encoder type, public API
    ├── config/             EncoderConfig (oxcf), validation
    ├── encoder/            VP9Comp-equivalent god-struct split
    │                       (decomposition is L-29; HLD-arch picks
    │                       "subsystem split with Arc-shared cm" — §3.2)
    ├── frame_loop/         vp9_get_compressed_data equivalent
    ├── partition/          choose_partitioning + nonrd_pick_partition
    ├── mode_decision/      vp9_pick_inter_mode + non-RD cost model,
    │                       Laplacian R-D tables (A2 §5.4 patch, R-07)
    ├── motion_est/         full-pel + sub-pel ME (A3)
    ├── rate_control/       1-pass CBR + VBR-RT, buffer model, drop
    ├── aq/                 cyclic_refresh (mode 3 only, D-05)
    ├── denoiser/           temporal denoiser + noise estimate
    ├── svc/                spatial + temporal SVC layer mgmt
    ├── speed_features/     RT cascade (A6 §5.7-§5.17, byte-exact)
    ├── threading/          tile-MT, row-MT, worker pool
    │                       (HLD-threading owns)
    └── bitstream_writer/   vp9_pack_bitstream equivalent
```

The encoder is **the** consumer of `rvp9-core`. Every shared kernel
the encoder needs to call (transform, intra-pred, loopfilter, quant,
range encoder, FRAME_CONTEXT cost LUTs) comes from `rvp9-core`.

---

## 3. Type-system overview

This section enumerates the load-bearing types that cross module
boundaries, where each lives, who owns it, and its synchronization
semantics. **Every entry cites Stage 1.** Synchronization claims for
worker threads anchor to 00-overview §3 (state graph). Detailed
ownership decisions for tile workers belong to **HLD-threading**; this
section commits only to the cross-crate / cross-module signatures.

### 3.1 Pixel types (load-bearing — see §7 for full policy)

| Type | Defined in | Owner | Sync |
|---|---|---|---|
| `Pixel = u8` | `rvp9_core::geom` (alias) | crate-public | n/a — scalar |
| `Coeff = i16` | `rvp9_core::transforms` | crate-public | n/a |
| `TranLow = i32` | `rvp9_core::transforms` | crate-public | n/a — sized for inverse-transform intermediate ranges. v1 is 8-bit so `tran_low_t` collapses to `i16` capacity but we keep `i32` to match libvpx exactly (D7) |

§7 owns the full pixel-type story including the v2 expansion path.

### 3.2 Frame buffers

| Type | Defined in | Owner | Sync |
|---|---|---|---|
| `FrameBuffer` | `rvp9_core::frame` | `FramePool` | content immutable while shared; per-plane raw-pointer aliasing into tile-disjoint slices is granted by `FramePool::lease_for_tile_write(&mut FrameBuffer, TileGeom) -> TileWriteToken` documented in HLD-threading |
| `RefCntBuffer` | `rvp9_core::frame` | `FramePool` (inner refcount) | `Arc<FrameBuffer>` underlying — `ref_count` field accessed only on the main thread (00-overview §3.1; A1 §7.12) |
| `FramePool` | `rvp9_core::frame::pool` | `Common` (decoder) / `EncoderCommon` (encoder) | 12 slots = 8 ref + 1 new + 3 scaled (A1 §2.8); public API exposes only the 8 ref slots, the 3 scaled slots are implementation detail per D-08 / A7 #22; pool itself is `Send` but not `Sync`; acquisition/release is single-threaded "main thread only" matching A1 §3.9 |
| `RefFrames` | `rvp9_core::frame` | per-frame transient on `Common` | snapshot of 3 reference indices for the current frame (`cm->frame_refs[3]`); read-only by tile workers (00-overview §3.1) |

**Ownership transfer to workers.** The Stage 1 invariant (00-overview
§3.6) is that workers may write only:

1. Their own `ThreadData`.
2. Their tile slice of `cpi->tile_data[t]`.
3. Tile-disjoint slices of `cm->above_context`,
   `cm->above_seg_context`, `cm->mi_grid_visible`, `cm->mi`,
   `cm->current_frame_seg_map`, `cm->frame_to_show->buffer` (its tile
   columns), `cpi->denoiser.running_avg_y[]` (its tile's pixels),
   `cpi->prev_partition`, `cpi->prev_segment_id`,
   `cpi->prev_variance_low`, `cpi->copied_frame_cnt`,
   `cpi->content_state_sb_fd`, `cpi->consec_zero_mv`,
   `cpi->segmentation_map`, `cpi->tile_tok[][]`, `cpi->tplist[][]`.

For `rvp9-core` HLD: frame buffers are `Arc<FrameBuffer>` with the
recycling pool managed by `FramePool` in `rvp9_core::frame::pool`.
Ownership transfers to tile workers via a raw-pointer cast under a
documented aliasing contract owned by **HLD-threading**. The contract
template is: `unsafe fn slice_for_tile<'a>(&'a self, tile: TileGeom)
-> TileFramePlanes<'a>` with a `SAFETY` block enumerating which planes
are split, by what stride, and the proof obligation that no two
overlapping calls are live concurrently. HLD-arch fixes only that:

- Reference frames passed *into* a worker are `&RefFrames` (immutable
  borrow for the duration of the encode).
- The output frame `cm->frame_to_show` is mutated tile-disjointly per
  the §5 unsafe policy.
- The `Arc` reference count is touched only on the main thread between
  frames (`vp9_update_reference_frames` — A1 §3.9). Tile workers do
  not bump it.

L-37 (Arc vs Rc) resolves to **Arc**: the encoder is required to be
`Send` and may be used from multiple host threads even though tile
workers are scoped, plus the `Arc` clone is the explicit
single-threaded discipline gate.

### 3.3 Motion vectors

| Type | Defined in | Owner | Sync |
|---|---|---|---|
| `MotionVector { row: i16, col: i16 }` | `rvp9_core::geom` | crate-public | Copy + Eq; no sync |
| `MvReference` | `rvp9_core::geom` | crate-public | enum of LAST/GOLDEN/ALTREF |

The encoder has additional motion-vector data structures (`MV_CANDIDATE`,
`int_mv`, `B_MODE_INFO`) which live in `rvp9-encoder::mode_decision`
and are not exposed across the crate boundary.

### 3.4 Probability contexts

| Type | Defined in | Owner | Sync |
|---|---|---|---|
| `FrameContext` | `rvp9_core::entropy` | `Common` (4-slot ring) | snapshot, `Clone`; mutated only between frames by `vp9_adapt_*_probs` (A1 §3.4 step 54, 00-overview §4.3) |
| `FrameCounts` | `rvp9_core::entropy` | per-worker | each worker owns its own `FrameCounts` (via `ThreadData`); merged back into `Common::counts` after sync via `accumulate_frame_counts` (scalar-only kernel, A6 §6) |
| `DefaultProbs` | `rvp9_core::entropy` | `static` | every probability default from A4 §5 (`default_coef_probs_*`, `vp9_pareto8_full`, MV trees, partition probs, skip probs, tx-size probs, segmentation probs, switchable interp probs); compile-time constants; D-14 (`vp9_prob_cost[256]`) included |
| `ProbCost` | `rvp9_core::entropy` | `static` | the 256-entry cost LUT (D-14, async A7 #3); without this every cost computation silently diverges |

Per L-09 / 00-overview §4.3: encoder and decoder both read
`FrameContext` (decoder for parsing, encoder for cost LUT derivation
in `vp9_initialize_rd_consts`). The encoder builds derived cost LUTs
(`inter_mode_cost`, `intra_uv_mode_cost`, `partition_cost`, etc.) from
the current `Common::fc` per-frame on the main thread; tile workers
read the cost LUTs, never `Common::fc` directly. The cost LUTs live in
`rvp9-encoder::mode_decision::cost_tables`, not in `rvp9-core`.

### 3.5 Range coder (BoolCoder)

| Type | Defined in | Owner | Sync |
|---|---|---|---|
| `BoolReader` | `rvp9_core::bitstream` | per-tile decoder state | non-Send during use; one per tile |
| `BoolWriter` | `rvp9_core::bitstream` | per-tile encoder token buffer | non-Send during use; one per tile |

A4 §1 / §3 covers the range coder. Encoder writes tokens into per-tile
buffers, then the main thread serializes them into the final
bitstream (A1 §3.4 step 49). Parallel pack rejected per L-35.

### 3.6 Encoder god-struct decomposition (L-29 resolution)

**Decision:** subsystem split with shared inner `Arc<Common>`. The
libvpx `VP9_COMP` (A1 §2.4) has roughly 120 fields organized in 22
logical groups; A1 enumerates them. Per L-29 / L-72 we **do not**
preserve a flat mega-struct.

Concrete shape (HLD-level; encoder LLD owns the field-by-field
distribution):

```text
Encoder
├── cfg: EncoderConfig                  (oxcf, frozen at construction)
├── common: Arc<Common>                 (rvp9_core::frame::Common —
│                                        BufferPool, fc, frame_contexts,
│                                        ref_frame_map, counts)
├── rc: RateControl                     (rvp9_encoder::rate_control —
│                                        cpi->rc, ~70 fields)
├── sf: SpeedFeatures                   (rvp9_encoder::speed_features —
│                                        per-frame snapshot, immutable
│                                        during a frame encode)
├── svc: SvcState                       (rvp9_encoder::svc, behind cfg
│                                        feature `svc`; otherwise unit type)
├── denoiser: Option<Denoiser>          (feature `denoiser`)
├── cyclic_refresh: CyclicRefresh       (AQ mode 3 only — D-05)
├── noise_estimate: NoiseEstimate
├── lookahead: Lookahead                (FIFO, A1 §2.9)
├── ext_rc: Option<Box<dyn RateControllerExt>>  (§8)
├── workers: WorkerPool                 (HLD-threading owns)
├── thread_data: Vec<ThreadData>        (per-worker; HLD-threading owns)
└── tile_data: TileDataMatrix           (per-tile; HLD-threading owns)
```

Rationale (00-overview §3.1, §7.2 HLD-arch scope):

- Subsystem split lets each subsystem be a Rust module with private
  state and a small public surface (read by the frame loop).
- `Arc<Common>` is shared with tile workers as `&Common` (no clones
  during encode); per A1 §7.12 / 00-overview §3.1, no worker writes to
  `Common` during a frame encode. The "8-slot ref map is immutable
  during encode" invariant becomes a `&Common` borrow.
- `RateControl`, `SpeedFeatures`, `CyclicRefresh`, `NoiseEstimate` are
  all owned by the encoder god-struct, mutated only on the main
  thread, and **snapshot** (cloned or borrowed) into worker thread
  data as needed (D-10 in HLD-threading).

Rejected: a god-struct named `Vp9Comp` that owns all subsystems flat.
The Stage 1 `VP9_COMP` is a god-struct because C has no namespace
mechanism; Rust does. L-29 / L-72 close on subsystem-split.

### 3.7 The TileWorkerContext (cross-cut placeholder for HLD-threading)

| Type | Defined in | Owner | Sync |
|---|---|---|---|
| `ThreadData` | `rvp9_encoder::threading` | one per worker | owned exclusively by the worker; lifetime matches the worker pool; mutated freely; merged back to god-struct state after sync (A6 §2.5 — value-copy-out protocol) |
| `TileData` | `rvp9_encoder::threading` | one per tile (column × row) | `thresh_freq_fact`, `mode_map`, `row_mt_sync`; per-tile writers; HLD-threading owns aliasing (A6 §2.x, 00-overview §3.4) |
| `TileGeom` | `rvp9_core::geom` | crate-public | tile column index, row index, plus the four pixel-coordinate bounds for the tile; immutable; `Copy` |

This is a stub. **HLD-threading** owns the full type-system shape of
worker / tile state. The fields and aliasing model are settled there
and any change there does not require revisiting HLD-arch.

---

## 4. Error model

### 4.1 `Result` policy

**All fallible operations return `Result<T, _>`.** No panics on bad
input. The error types are crate-scoped.

```text
rvp9_core::Error          (parsing, decoding, kernel-level)
rvp9_encoder::EncodeError (encoder-only failures)
rvp9_decoder::DecodeError (decoder-only failures)
```

Per L-09 / 00-overview §4.10: this replaces libvpx's `setjmp`/`longjmp`
panic mechanism (D-09 RESOLVED in HLD-arch — see below). Workers in
libvpx C never call `vpx_internal_error`; OOM and parser errors occur
only on the main thread at frame-init time. In Rust we propagate
`Result` end-to-end without longjmp. Frame loop returns
`Result<Option<EncodedPacket>, EncodeError>` so a graceful frame-drop
(A5: `vp9_rc_drop_frame`) is `Ok(None)`, a real failure is `Err(...)`.

### 4.2 Error type design

`rvp9-core::Error` is the broadest type:

```text
enum Error {
    BitstreamTruncated { offset: usize, needed: usize },
    InvalidProfile(u8),
    UnsupportedProfile { profile: u8 },          // e.g. 2/3 in v1 (HBD)
    InvalidFrameSize { width: u32, height: u32, reason: &'static str },
    InvalidProbability { context: &'static str },
    ReferenceUnavailable { slot: u8 },
    AllocationFailed { what: &'static str, bytes: usize },
    InternalInvariant(&'static str),
}
```

Encoder and decoder error types `From`-convert from `rvp9_core::Error`
and add their own variants (e.g., `EncodeError::RateControlOvershoot`,
`DecodeError::CorruptTileData`).

Rules:

- `Error` is `Send + Sync + 'static`. Workers must be able to surface
  errors back to the main thread.
- `Error` implements `std::error::Error` and `Display`.
- All `&'static str` payloads name a specific call site to make the
  cause line up with the divergence ledger.
- No `Box<dyn Error>` in public APIs.

### 4.3 Panic policy

**Panics are reserved for genuine programmer errors.**

Permitted panic sites:

1. Out-of-bounds index that violates an internal invariant (use
   `debug_assert!` for the check, then index directly in release).
2. `unreachable!()` for arms that the type system cannot rule out but
   the algorithm guarantees never to hit. Stage 1 §3.4 step 13 (KF
   reset) and similar discrete enum dispatches are candidates.
3. `unwrap()` on a `Result<T, Infallible>` (no real fallibility).
4. Stage 1-level "this is a debug-only check": Miri builds and
   `cfg(debug_assertions)`.

**Forbidden panic sites:**

1. Anywhere reachable from `Decoder::decode_frame` on malformed input.
   Malformed bitstream → `Err(Error::Bitstream*)`.
2. Anywhere reachable from `Encoder::encode_frame` on legal user
   input (any frame in the configured profile/dimensions). Rate-control
   overshoot, frame drop, AQ map saturation: all return `Result`.
3. Tile workers must not panic on encoded content. Worker panics are
   caught by the worker pool join logic (HLD-threading owns the
   `std::thread::scope` join semantics) and converted to a
   `EncodeError::WorkerCrashed`.

`panic = unwind` (the default). We do not switch to `panic = abort`
because the worker join logic depends on unwind to catch worker panics
gracefully. (HLD-threading confirms.)

D-09 (setjmp/longjmp Rust error model) **RESOLVED** here: `Result<T,
EncodeError>` end-to-end; no global error machinery; worker panics
caught at join.

### 4.4 Assertion policy

- `assert!` — runtime check that holds in both debug and release; use
  for invariants that are cheap to check and load-bearing for memory
  safety. Example: tile geometry bounds before pointer casts (§5).
- `debug_assert!` — debug-only checks for hot paths.
- `assert_eq!` / `assert_ne!` — only in tests and golden-file checks.

Stage 1 derived asserts:

- `assert!(log2_tile_rows == 0)` per L-62 in encoder dispatch.
- `debug_assert!(block_height == block_width)` per L-52 in
  `set_vt_partitioning`.

---

## 5. `unsafe` policy (load-bearing)

D6 commits us to libvpx-mirrored threading: tile workers sharing
slices of large buffers under a documented "main thread does
everything else" aliasing contract (00-overview §3.6, A1 §7.12, A6
§3). This means **substantial** `unsafe` in the encoder. The decoder
needs much less but is not zero (SIMD intrinsics). This section sets
the rules.

### 5.1 Workspace-level configuration

`Cargo.toml` already sets `unsafe_op_in_unsafe_fn = "warn"` workspace
wide. We promote this to **`deny`** for the encoder crate in v1:

```toml
[lints.rust]
unsafe_op_in_unsafe_fn = "deny"
```

This forces every `unsafe` operation inside an `unsafe fn` to be
inside an explicit nested `unsafe { ... }` block, which gives the
safety comment its anchor.

We do **not** use `#![forbid(unsafe_code)]` anywhere — it would break
the entire encoder. We **do** restrict `unsafe` to the modules listed
below.

### 5.2 Where `unsafe` is permitted

**Permitted, with a documented safety contract per call site:**

1. **`rvp9-core::simd::x86_64`, `rvp9-core::simd::aarch64`** — SIMD
   intrinsics from `core::arch::*`. Wrapper functions document the
   target-feature precondition (e.g. `// SAFETY: caller has verified
   the CPU supports AVX2 via runtime detection in simd::dispatch`).

2. **`rvp9-core::simd::dispatch`** — function-pointer table
   construction. The dispatch table itself is initialised once at
   process start; the call sites do `unsafe { fn_ptr(args) }` only
   after the CPU-feature gate.

3. **`rvp9-encoder::threading`** — the entire tile / row worker
   subsystem. Specifically:
   - Raw-pointer casts to split frame planes / above-context buffers
     into tile-disjoint slices. Each cast site has a `SAFETY:` block
     listing: which buffer, which slice bounds, which thread reads or
     writes, what guarantees disjointness, what guarantees no other
     thread aliases the same bytes during the lifetime of the cast.
   - `UnsafeCell` use for the few fields where libvpx has workers
     mutate without locking (per 00-overview §3.6 — `denoiser.running_avg_y[]`,
     `segmentation_map` per-block segment_id writes, `tile_data` per-tile
     state). HLD-threading owns the precise list.
   - `Send` / `Sync` implementations on wrapper types around raw
     pointers; each implementation must justify the safety in the
     `unsafe impl` comment.

4. **`rvp9-encoder::motion_estimation`** — only for SIMD-adjacent
   pointer arithmetic into reference frame planes. Same comment
   template as §5.5.

5. **`rvp9-core::bitstream`** — minimal. The boolean decoder/encoder
   inner loop benefits from raw-pointer reads of the input buffer for
   performance. Permitted but each call site documented; the slow
   reference paths in `simd::scalar` remain safe.

6. **`rvp9-core::frame`** — buffer allocation and the
   `Arc::clone` / `Arc::strong_count` machinery is safe code. The
   only `unsafe` is the slice-split function used by HLD-threading
   when handing tile slices to workers.

### 5.3 Where `unsafe` is forbidden

Forbidden everywhere except the modules listed in §5.2:

- `rvp9-core::entropy`, `rvp9-core::transforms`, `rvp9-core::quant`,
  `rvp9-core::intra_pred`, `rvp9-core::loopfilter`, `rvp9-core::geom`,
  `rvp9-core::color`, `rvp9-core::ratectrl_ext`.
- `rvp9-decoder` non-SIMD code.
- `rvp9-encoder` outside the threading + motion-estimation modules
  listed in §5.2.

These modules will be Miri-clean in CI (Stage 5 hardening).

### 5.4 Mandatory safety-comment template

Every `unsafe { ... }` block — whether expression or function — must
have a `// SAFETY: ...` comment **immediately above** that addresses
each of the following points (omit a point only if it is genuinely
not applicable, in which case the comment says so):

```rust
// SAFETY:
// - Pointer provenance: <where the pointer came from; for slice splits,
//   cite the function that produced the slice and its lifetime>
// - Bounds: <how we know the pointer + offset is within the
//   allocation; cite the assert! / debug_assert!>
// - Alignment: <Pixel is u8 in v1 so alignment is trivial; SIMD reads
//   require the wider alignment, document the source guarantee>
// - Aliasing: <which other threads or call sites could touch this
//   memory; cite the invariant proving disjointness; for tile splits,
//   cite the TileGeom that proves non-overlap>
// - Lifetime: <how we know the underlying allocation outlives this
//   pointer>
// - CPU feature (SIMD only): <which runtime check guarantees the
//   target-feature precondition>
```

The Stage 5 audit (PLAN.md §6 Stage 5 task 4) reviews every `unsafe`
block against this template.

### 5.5 Concrete templates

**Template A — tile-disjoint slice split** (used by `rvp9_encoder::threading`):

```rust
// SAFETY:
// - Provenance: `planes` was produced by `FramePool::lease_for_frame()`
//   on the main thread, which returned exclusive access.
// - Bounds: `tile.x_start`, `tile.x_end` are in [0, plane_width); the
//   debug_assert! above this block verifies.
// - Alignment: u8 plane, alignment is 1; SIMD wrappers handle their
//   own alignment internally.
// - Aliasing: tile columns are pixel-disjoint (00-overview §3.6;
//   A1 §7.12); the `TileSplitGuard` keeps the parent borrow alive
//   for the duration of the worker's scoped thread.
// - Lifetime: parent `&'a mut FrameBuffer` outlives this slice via
//   the `'a` lifetime on the returned `TileFramePlanes<'a>`.
// - CPU feature: n/a (no SIMD intrinsics here; kernel callees gate
//   their own features).
unsafe { plane_ptr.add(offset) }
```

**Template B — SIMD intrinsic** (used by `rvp9_core::simd::x86_64`):

```rust
// SAFETY:
// - Provenance: input/output slices from the caller, validated by
//   the dispatch table to be at least 4 bytes long (TX_4X4 minimum).
// - Bounds: caller's bounds-check on the slice length is assumed by
//   contract; the wrapping `transforms::idct4x4_avx2` checks the
//   slice length.
// - Alignment: AVX2 256-bit load requires 32-byte alignment for
//   `_mm256_load_si256`; we use the unaligned `_mm256_loadu_si256`
//   variant so any alignment is OK.
// - Aliasing: input and output may not overlap; this is a function
//   contract enforced by the caller (transforms always have separate
//   src/dst buffers per libvpx convention).
// - Lifetime: returns once intrinsic completes; no stored pointer.
// - CPU feature: runtime gated by simd::dispatch on AVX2 detection.
unsafe { _mm256_loadu_si256(src.as_ptr() as *const __m256i) }
```

### 5.6 Process for adding new `unsafe`

1. The author writes the safety comment first, then the code.
2. PR review checks the comment for each of the six points.
3. The divergence ledger gets an entry only if the unsafe block
   diverges semantically from libvpx (most don't — they faithfully
   mirror C pointer arithmetic with an explicit Rust contract).
4. Miri is run on the affected non-SIMD path before merge.
5. Stage 5 final audit reviews every entry.

### 5.7 `UnsafeCell` policy

`UnsafeCell` is used **only** for fields that:

1. Are mutated by tile workers concurrently and the workers' write
   regions are provably disjoint per 00-overview §3.6.
2. Cannot be modelled as a slice split (e.g. interior-mutable counter
   that the worker increments).

Every `UnsafeCell` field has its own `SAFETY` paragraph describing the
provably-disjoint regions or the synchronization that protects it.

`UnsafeCell` is **forbidden** in `rvp9-core` apart from the SIMD
dispatch table init (which is `OnceLock`-based, no UnsafeCell needed).

---

## 6. Integer-width policy

D7 commits us to mirror libvpx integer widths exactly. PLAN.md D7
rationale: same as D6 (maximizes byte-parity tractability; idiomatize
post-v1).

### 6.1 The mirror rule

For every libvpx integer-typed variable that crosses an algorithm
boundary (function call, struct field, intermediate result fed back),
the Rust port uses the same fixed-width integer type:

| libvpx | rvp9-2 |
|---|---|
| `uint8_t` | `u8` |
| `int8_t` | `i8` |
| `uint16_t` | `u16` |
| `int16_t` | `i16` |
| `uint32_t` | `u32` |
| `int32_t` | `i32` |
| `int64_t` | `i64` |
| `uint64_t` | `u64` |
| `int` (the unqualified C int) | `i32` — libvpx is built on 64-bit platforms where `int` is 32-bit; we pin it |
| `unsigned int` | `u32` — same reason |
| `size_t` | `usize` (Rust's native pointer-sized) — **except** when libvpx stores a `size_t` in a struct that is serialised or memcmp'd; then we use `u64` to pin the layout |
| `ptrdiff_t` | `isize` for stride values; **except** when a libvpx stride is stored in a struct — then `i32` to match the C field width |

This isn't a "choose what's idiomatic"; for every field that touches
a transform, quant, or arithmetic coder, we use libvpx's type.

### 6.2 The `wrapping_*` convention

Rust checks integer overflow in debug and wraps in release. C signed
overflow is UB but libvpx assumes two's-complement wrap. For
byte-parity we **wrap in both debug and release** at every site where
libvpx's arithmetic would wrap:

- Every arithmetic operation mirroring a libvpx expression uses the
  explicit `wrapping_*` method (`wrapping_add`, `wrapping_mul`,
  `wrapping_sub`, `wrapping_neg`). Plain `+`/`-`/`*` is reserved for
  the rare sites where libvpx clearly cannot overflow (operand bound
  by table size, block geometry, etc.), and the comment must note it.
- Shifts (`<<`, `>>`) do not wrap. Rust panics in debug if the count
  exceeds the bit width; where libvpx shifts by a bounded runtime
  value we add a `debug_assert!` on the count, then shift normally.
- Comparisons do not wrap and need no special treatment. Float ops
  are unchanged (Rust IEEE-754 matches C).

### 6.3 Bit-packed fields

Where libvpx uses C bit-fields (`int x : 1;`), we use the smallest
fixed-width integer that fits and the comment documents the bit width.
A `:1` becomes a `bool` if its semantic is boolean and `u8` masked to
1 if it is later combined with other small fields into a bit-packed
representation for the bitstream writer (A4 §3).

### 6.4 Quant intermediates

The transforms use a wider type for inverse-transform intermediates
(`tran_low_t` in libvpx). v1 is 8-bit only (D-01 RESOLVED), and the
libvpx default in non-HBD builds is `int16_t`; libvpx has a
build-time switch to use `int32_t` for safety analyses. Per D7 we
**pin `TranLow = i32`** so the v2 HBD path slots in without
re-typing every kernel signature. The cost is a 2× memory footprint
in transform intermediates which is negligible vs the per-frame total.

### 6.5 What we do *not* mirror

- libvpx's `INT64_MAX` / `INT_MAX` constants used as sentinels become
  `i64::MAX` / `i32::MAX`. The values are identical on every target;
  we use the Rust constant for readability.
- libvpx's `ROUND_POWER_OF_TWO(value, n)` is a macro
  `(value + (1 << (n - 1))) >> n`. We provide an inline function
  `round_power_of_two(v: i32, n: u32) -> i32` that contains the
  `wrapping_add` and the shift, used everywhere libvpx uses the
  macro. The function lives in `rvp9_core::transforms::common`.

---

## 7. 8-bit-only pixel-type policy (load-bearing)

This section replaces the now-defunct `03-bit-depth.md` per the Stage 2
entry scope. D-01 RESOLVED: 8-bit (Profile 0) only in v1; HBD deferred
to Stage 6+.

### 7.1 Type aliases

```text
rvp9_core::geom::Pixel        = u8       // luma + chroma sample
rvp9_core::geom::Coeff        = i16      // post-quant coefficient
rvp9_core::transforms::TranLow = i32      // inverse-transform intermediate
rvp9_core::geom::PixelWide    = u16      // reserved, see §7.4 below
```

The aliases live in `rvp9_core::geom` and are re-exported at the
crate root. Every kernel signature uses the alias, not `u8`:

```rust
pub fn idct8x8_scalar(
    coeffs: &[Coeff; 64],
    dst: &mut [Pixel],
    stride: usize,
);
```

`Pixel = u8` and the alias is everywhere a buffer of luma/chroma is
passed between modules.

### 7.2 The "no generic over bit depth" decision

We **do not** make kernels generic over the pixel type in v1. The
options considered:

1. **Hardcode `u8`** (chosen). Every kernel is `fn idct8x8(...)` taking
   `&[u8]`. Smallest code; smallest binary; smallest type-error
   surface area.

2. **Generic `fn idct8x8<P: PixelLike>(...)`**. Requires a `PixelLike`
   trait abstracting over `u8` / `u16` arithmetic. **Rejected** for
   v1: every kernel would need to be generic; SIMD intrinsics differ
   per bit depth (no shared instruction); monomorphisation balloons
   the binary; trait bounds infect every call site.

3. **Macro-generated kernel pairs** (libvpx's approach, via
   `vpx_dsp_rtcd_defs.pl`). **Rejected** for v1: the value
   proposition is realising a second bit depth, which is out of scope.
   Reconsider in Stage 6+.

**Stage 1 anchor:** A7 challenge #11 (CONFIG_VP9_HIGHBITDEPTH leakage
in real-time paths) → D-01 RESOLVED. 00-overview §7.6 "out of scope"
list includes HBD.

### 7.3 Profile gating

The decoder accepts Profile 0 only in v1. Profiles 1/2/3 return
`Error::UnsupportedProfile { profile }` at frame-header parse time.
The encoder constructor refuses any non-zero profile in `EncoderConfig`
with an `EncodeError::UnsupportedConfiguration`.

This is a deliberate, documented restriction. Conformance vectors for
Profile 0 are the only ones we run in Stage 3 (PLAN.md Stage 3 exit
criterion: "100% pass on official VP9 conformance test vectors"
implicitly scopes to Profile 0 per D-01).

### 7.4 The `PixelWide = u16` reservation

`PixelWide = u16` is reserved at the type level but **not used** in
v1. Its presence in `rvp9_core::geom` declares the v2 expansion path
without committing any code to it. The alias is `#[doc(hidden)]` in
v1 to avoid surfacing it in rustdoc.

### 7.5 V2 expansion path

When Stage 6+ adds HBD, two options remain on the table; the v1 design
does not foreclose either:

1. **Generic kernels with a `PixelLike` trait.** Add the trait,
   make all kernels generic. The current API signatures change from
   `&[u8]` to `&[T]` where `T: PixelLike`. The encoder + decoder type
   `Pixel` becomes a const-generic or type-parameter parameter on the
   `Encoder` / `Decoder` types. Cost: every call site needs the type
   parameter or a top-level monomorphisation. The Rust binary grows
   per monomorphisation.

2. **Compile-time bit-depth selection.** A Cargo feature
   `pixel-depth-10` swaps the `Pixel` alias from `u8` to `u16`
   conditionally; kernels stay non-generic but a 10-bit build is a
   separate binary. Closer to libvpx's `CONFIG_VP9_HIGHBITDEPTH`
   approach. Cost: cannot encode/decode both depths in the same
   process binary.

3. **Type-level configuration on Encoder / Decoder.** Both `u8` and
   `u16` paths coexist; the `Encoder` type itself carries a
   `Pixel` const-generic. Most flexible; most monomorphisation.

The v2 decision is deferred. The v1 design avoids any choice that
forecloses a path. Concretely:

- Kernel signatures take `&[Pixel]` (the alias) rather than `&[u8]`
  literally. Renaming `Pixel` is a v2 search-and-replace, not a v1
  bug.
- The `Coeff` and `TranLow` types are already wide enough for 10-bit
  (i16 / i32) so the intermediate types do not change in v2.
- The encoder god-struct decomposition (§3.6) is parametric in nothing,
  but it does **not** hardcode `u8` in field types; everything goes
  through `Pixel`.

### 7.6 SIMD implications

Per-arch SIMD intrinsics for `u8` are distinct from `u16` (e.g.
`_mm256_avg_epu8` vs `_mm256_avg_epu16`). v1's SIMD code paths
exclusively use `u8` intrinsics. The v2 HBD path will add a parallel
set of `u16` intrinsics under a feature gate. **HLD-simd** owns the
dispatch organization; the table layout is per-kernel and per-pixel-type,
so v2 adds a row but does not restructure existing rows.

### 7.7 Quantizer & dequant gating

A4 §2 + A5 §2 show that quant tables have separate 8-bit and HBD
variants in libvpx. v1 ships only the 8-bit tables. The `quant` module
in `rvp9-core` documents this and refuses HBD inputs at compile time
(via `Pixel = u8` propagation; an HBD quantizer cannot be constructed
because there is no HBD pixel alias visible).

---

## 8. EXT_RATECTRL plug-in interface

R-06 RESOLVED: in v1. WebRTC integration requires it. The interface is
defined in `rvp9-core::ratectrl_ext` so that downstream crates can
depend on a small surface without pulling in the full encoder.

### 8.1 Reference

libvpx exposes the plug-in via `vp9_ext_ratectrl.h` (a callback table)
plumbed into `cpi->ext_ratectrl` (A1 §2.4, §3.4 step 20, §7.7;
00-overview §3.1 row `cpi->ext_ratectrl`). The libvpx surface is
roughly:

```
extrc_init / extrc_create_model / extrc_send_firstpass_stats /
extrc_send_tpl_gop_stats / extrc_get_encodeframe_decision /
extrc_update_encodeframe_result / extrc_get_key_frame_decision /
extrc_get_gop_decision / extrc_get_frame_rdmult / extrc_delete
```

For real-time / WebRTC v1 we only need a subset: per-frame Q
decision, per-frame update, frame-drop decision, and lifecycle.

### 8.2 Rust trait surface

```text
pub trait RateControllerExt: Send {
    /// Called once when the encoder is constructed with an external
    /// controller. Implementations cache config they need.
    fn init(&mut self, cfg: &ExtRcConfig) -> Result<(), ExtRcError>;

    /// Called per frame before encode. Returns the controller's
    /// desired q-index and whether to drop the frame.
    fn get_encodeframe_decision(
        &mut self,
        info: &ExtRcFrameInfo,
    ) -> Result<ExtRcDecision, ExtRcError>;

    /// Called per frame after encode + pack, with actual bitstream
    /// stats. Allows the controller to update its internal state.
    fn update_encodeframe_result(
        &mut self,
        result: &ExtRcEncodeResult,
    ) -> Result<(), ExtRcError>;

    /// Called when the encoder shuts down. Allows the controller to
    /// flush state, write logs, etc.
    fn shutdown(&mut self) -> Result<(), ExtRcError> { Ok(()) }
}
```

Companion data types:

```text
pub struct ExtRcConfig { ... }          // initial config bundle
pub struct ExtRcFrameInfo { ... }       // per-frame pre-encode info
pub struct ExtRcDecision { ... }        // q-index, drop, GOP hints
pub struct ExtRcEncodeResult { ... }    // actual bytes, q used,
                                        // overshoot signals
pub enum ExtRcError { ... }             // controller-side failure
```

Full per-field shape belongs in LLD-rate-control. HLD-arch fixes only
that:

- The trait is `Send` (controller may be invoked from the
  encoder's main thread, which is not necessarily the user's thread).
- The trait is **not** `Sync` — callers must serialise their own
  access. The encoder calls it strictly single-threaded from the main
  frame loop.
- The trait is object-safe (no generics on its methods). The encoder
  holds `Box<dyn RateControllerExt>` in `Encoder::ext_rc`.
- Returning an `ExtRcError` from any method aborts the current frame
  with `EncodeError::ExternalRateControl(err)`.

### 8.3 How it slots into rate control

Without coupling to LLD-rate-control internals: the encoder's
`rate_control` module checks `Encoder::ext_rc.is_some()` at three
points per frame:

1. **Before encode.** If present, call
   `get_encodeframe_decision()`; use the returned q-index in lieu of
   the internal `vp9_rc_pick_q_and_bounds`. If the decision says
   drop, treat the frame as dropped (same code path as
   `vp9_rc_drop_frame` returning true).
2. **After encode + pack.** Call `update_encodeframe_result()` with
   the actual byte count, q used, and overshoot signals. The internal
   `vp9_rc_postencode_update` still runs (it manages internal
   buffer-model state that the encoder uses for non-RC heuristics like
   `rc.high_source_sad`), but the *bitrate-allocation* decisions are
   the external controller's.
3. **On encoder shutdown.** Call `shutdown()` on the controller.

When `Encoder::ext_rc.is_none()`, the encoder uses its internal
1-pass CBR / VBR-RT path (PLAN.md §2). The presence of `ext_rc` is the
toggle.

### 8.4 What lives in `rvp9-core` vs `rvp9-encoder`

`rvp9-core::ratectrl_ext`:

- the `RateControllerExt` trait definition,
- the public data types (`ExtRcConfig`, `ExtRcFrameInfo`,
  `ExtRcDecision`, `ExtRcEncodeResult`, `ExtRcError`),
- documentation and usage examples.

`rvp9-encoder::rate_control::ext`:

- the integration points that call into the trait,
- the bridging to internal `RateControl` state where needed (e.g.
  populating `ExtRcFrameInfo` from the encoder's state).

A WebRTC integration crate (downstream of this workspace) depends on
`rvp9-core` only and provides an `impl RateControllerExt for
WebRtcController { ... }`. The bridge crate then plugs this into
`Encoder::with_external_rate_control(impl)`.

### 8.5 Feature gating

The trait is always compiled in `rvp9-core` (no feature gate; small
surface, no extra dependencies). The encoder's integration code in
`rvp9-encoder::rate_control::ext` is behind a feature flag
`ext-ratectrl` which is **on by default** but can be turned off for
size-sensitive builds (R-06 RESOLVED: "mechanical work").

---

## 9. Build-time configuration

The Cargo feature matrix is small. The principle: **anything that
costs binary size or constrains the public API must be a feature
flag**; anything that is enabled-by-default and runtime-toggled is
always-on.

### 9.1 `rvp9-core` features

| Feature | Default | Effect |
|---|---|---|
| `simd-scalar` | on | always-on scalar fallback for every kernel; cannot be disabled (it is the correctness baseline) |
| `simd-sse2` | on (x86_64 only) | SSE2 intrinsics module compiled in |
| `simd-sse41` | on (x86_64 only) | SSE4.1 intrinsics module compiled in |
| `simd-avx2` | on (x86_64 only) | AVX2 intrinsics module compiled in |
| `simd-neon` | on (aarch64 only) | NEON intrinsics module compiled in |
| `no-simd` | off | disables every per-arch SIMD module; forces scalar path. For platform-portable builds, debugging, and Miri runs. |

The per-arch features are mutually compatible. Runtime dispatch
(HLD-simd / D-02) picks the best path the CPU supports at startup.

### 9.2 `rvp9-encoder` features

| Feature | Default | Effect |
|---|---|---|
| `denoiser` | on | enables the temporal denoiser + noise estimate modules; PLAN.md §2 requirement |
| `svc` | on | enables the spatial + temporal SVC pipeline; PLAN.md §2 requirement |
| `ext-ratectrl` | on | enables the external rate-controller plug-in integration (§8); R-06 RESOLVED |
| `aq-cyclic-refresh` | on | enables AQ mode 3 (the only mode in v1, D-05 RESOLVED). Off → encoder rejects `aq_mode = CyclicRefresh` in config. |
| `temporal-svc-only` | off | disables spatial-SVC pipeline; useful for early Stage 4d builds that don't need spatial. Stage 4e re-enables. |
| `byte-parity-debug` | off | enables extra invariant checks (assert that `wrapping_*` operations did not actually wrap; verify worker-mutation field invariants; etc.). For diff-testing against libvpx. |

PLAN.md §2 requirements (denoiser, SVC, AQ) are on by default. They
can be turned off in a build that wants a minimal encoder (e.g. a
specific WebRTC integration that does its own denoise). Turning them
off cuts compile time and binary size by a real amount (denoiser is
~3-5 KLOC, SVC is ~40-50% additional encoder complexity per
00-overview S0-2).

### 9.3 `rvp9-decoder` features

**None in v1.** The decoder must always handle every conforming
bitstream. Profile-0 only is enforced at runtime, not at compile time.

### 9.4 What is **not** a feature flag

- **HBD support** — v1 is 8-bit only (D-01, §7); v2 expansion path is
  deferred per §7.5.
- **Profile selection** — Profile 0 only, always.
- **libvpx `CONFIG_*` knobs** (RATE_CTRL, NON_GREEDY_MV,
  INTERNAL_STATS, BETTER_HW_COMPATIBILITY, REALTIME_ONLY) — all
  dropped per L-33; non-RT paths are not compiled. Divergence-ledger
  records each drop.
- **`error_resilient_mode` / `frame_parallel_decoding_mode`** —
  runtime bitstream header bits; honoured by the decoder, set by the
  encoder per user config. No flag.
- **Tile threading** — always available; thread count is a runtime
  config (D6 + D-03).
- **Frame-parallel encoding (D-04)** — deferred to v2.

### 9.5 Cross-feature constraints + MSRV

- `simd-scalar` is always on; `no-simd` forces it even on
  x86_64/aarch64.
- `byte-parity-debug` is incompatible with Miri (wrapping invariants
  panic on wrap).
- Workspace MSRV is `rust-version = "1.95"`, edition `2024`. No
  nightly features; `core::arch` SIMD, `unsafe_op_in_unsafe_fn`, and
  `std::thread::scope` are all stable at MSRV.

---

## 10. Documentation policy

### 10.1 rustdoc requirements

`missing_docs = "warn"` is workspace-wide. **HLD-arch elevates this to
`deny`** for `rvp9-core` and `rvp9-encoder` public APIs (decoder
follows in Stage 3 once stabilised; CLI/bench/harness stay at `warn`
since they are tooling).

### 10.2 Doc comment shape

- Public types: 1-line summary + paragraph on ownership / lifetime /
  thread-safety, plus an example for types used directly.
- Public functions: 1-line summary + `# Errors`, `# Panics`,
  `# Safety` sections as appropriate. Every fallible function has
  `# Errors`; every `unsafe fn` has `# Safety`.
- Internal items (`pub(crate)` and below): 1-line summary at
  minimum — enforced by code-review standard, not the lint.

### 10.3 Library-level examples

`rvp9-encoder` and `rvp9-decoder` each ship a top-level rustdoc example
showing basic usage (decode an `.ivf` to YUV; encode YUV at 1 Mbps
CBR). Doc examples must compile and run under `cargo test --doc`.
Longer examples live in `crates/rvp9-cli/examples/`:

- `encode_realtime.rs` — 720p30 CBR encoder.
- `decode_to_yuv.rs` — minimal decoder.
- `encode_svc.rs` — 3-temporal-layer SVC.
- `encode_with_ext_rc.rs` — `RateControllerExt` integration (§8).

### 10.4 Cross-references and hidden items

Module-level rustdoc may cite Stage 1 specialist docs in prose;
function-level docs cite libvpx file:line only when the function
mirrors a specific libvpx algorithm. The `simd::dispatch` internals
and per-arch intrinsic submodules (`simd::x86_64`, `simd::aarch64`)
are `#[doc(hidden)]`. Internal `unsafe` helpers used by HLD-threading
are `pub(crate)`, code-commented but not rustdoc'd.

---

## 11. Open questions

These items are not blocking Stage 2 progress on HLD-threading,
HLD-simd, or HLD-svc. HLD-arch states its best assumption for each
and the named HLD picks them up.

For each item below, HLD-arch states its best assumption and the
named sibling HLD or LLD confirms or rejects without blocking Stage 2
progress.

| # | Item (anchor) | HLD-arch assumption | Owner |
|---|---|---|---|
| 11.1 | Worker-pool sizing (L-31) | Eager spawn at `Encoder::new`; pool has `cfg.max_threads` workers; lazy rejected for v1 (real-time p99 budget). | HLD-threading |
| 11.2 | Tile-thread primitives (D-03) | `std::thread::scope` + hand-rolled `Mutex<JobQueue> + Condvar` for row-MT (A6 §7.3). | HLD-threading |
| 11.3 | SpeedFeatures across workers (D-10) | Snapshot-per-frame, immutable `&SpeedFeatures` borrow per worker (A6 §7.1, 00-overview §3.1). | HLD-threading |
| 11.4 | ThreadData::mb aliasing (D-11) | Value-copy of `MACROBLOCK` per worker on launch; merge back via `accumulate_rd_opt` and `accumulate_frame_counts` (A6 §2.5, §3.3). | HLD-threading |
| 11.5 | SIMD dispatch table (D-02) | Hand-written tables per kernel family, one `OnceLock<DispatchTable>` each, populated at startup with runtime feature detection. | HLD-simd |
| 11.6 | Decoder frame-pool public API (D-08) | Decoder exposes only the 8 ref slots; the 3 scaled slots are encoder-internal (A7 #22). | LLD-frame-buffers |
| 11.7 | SVC ref-slot map (L-36, L-58, L-61) | 8-slot map shared across spatial + temporal layers; pool sized to top-resolution layer; per-layer RC swaps via `restore/save_layer_context`. `force_max_q` (L-74) and cross-layer drop (L-75) owned by HLD-svc. | HLD-svc |
| 11.8 | EXT_RATECTRL exact field shape | Public data types in §8.2 are field-by-field decided by LLD-rate-control; HLD-arch fixes only trait signatures, object-safety, `Send`-but-not-`Sync`, and module location. | LLD-rate-control |
| 11.9 | Float determinism in RC (R-04) | Pinned `log10`/`pow` impl (vendored libm or the `libm` crate) so diff-tests reproduce across hosts. | LLD-rate-control |
| 11.10 | `vp9_encodedframe_overshoot` (L-19, L-74) | Single `overshoot_recovery()` function in `rvp9_encoder::frame_loop` mirroring libvpx byte-for-byte. Five-mutation catalogue from async A7 #7 must be reproduced. | LLD-rate-control + LLD-frame-loop |

---

## 12. References

| Doc | Location | Used for |
|---|---|---|
| PLAN.md | `D:/language/rvp9-2/PLAN.md` | locked decisions D1-D9, success criteria, stage plan |
| 00-overview.md | `D:/language/rvp9-2/notes/vp9-analysis/00-overview.md` | Stage 1 synthesis; state graph (§3); cross-cuts (§4); A7 disposition (§5); Stage 2 entry briefing (§7) |
| 01-architecture.md | `D:/language/rvp9-2/notes/vp9-analysis/01-architecture.md` | VP9_COMP god struct; encoder lifecycle; BufferPool; reference rotation |
| 02-encoder-core.md | `D:/language/rvp9-2/notes/vp9-analysis/02-encoder-core.md` | non-RD mode decision; Laplacian R-D tables (R-07) |
| 03-motion-estimation.md | `D:/language/rvp9-2/notes/vp9-analysis/03-motion-estimation.md` | ME algorithms; SAD/variance dispatch |
| 04-bitstream-entropy.md | `D:/language/rvp9-2/notes/vp9-analysis/04-bitstream-entropy.md` | bitstream format; probability tables; range coder |
| 05-rate-control.md | `D:/language/rvp9-2/notes/vp9-analysis/05-rate-control.md` | RC structures; denoiser; AQ; ext_ratectrl plumbing |
| 06-threading-svc.md | `D:/language/rvp9-2/notes/vp9-analysis/06-threading-svc.md` | threading; SVC; speed features |
| decisions-needed.md | `D:/language/rvp9-2/notes/vp9-analysis/decisions-needed.md` | Stage 2 entry RESOLVED block; D / L / R items |
| Cargo.toml (workspace) | `D:/language/rvp9-2/Cargo.toml` | workspace layout, lints, profiles, MSRV |

---

*Document version: 0.1.0. HLD-arch — Stage 2 load-bearing top-level
architecture HLD. Stage 2 cross-HLD dependencies (HLD-threading,
HLD-simd, HLD-svc) anchor to this document; the open questions in §11
state HLD-arch's best assumption for each and pass to the named
sibling HLD for confirmation.*
