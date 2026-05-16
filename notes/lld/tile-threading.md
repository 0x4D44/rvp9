# LLD: tile-threading

**Stage 2 LLD.** Implementation-level design for the rvp9 encoder's
tile-MT + row-MT worker pool. This is the load-bearing concretisation of
`notes/hld/01-threading.md` (HLD-threading): the HLD picks strategies
A-F per field; this LLD names the structs, gives their field-by-field
Rust shapes, annotates `Send` / `Sync` / lifetime, lays out every
`unsafe` block with its mandatory SAFETY comment, and writes the
algorithmic pseudocode for tile-MT, row-MT, and the work-stealing job
queue.

**Authors:** Stage 2 (this run).
**Date:** 2026-05-16.
**Reference implementation:** `third_party/libvpx/` at v1.14.1
(`12f3a2a`). All `third_party/libvpx/<path>:<line>` citations resolve
there.
**Primary specialist source:** `notes/vp9-analysis/06-threading-svc.md`
(A6, 2002 lines).
**Secondary source:** `notes/vp9-analysis/01-architecture.md` (A1) for
the encoder lifecycle integration (`vp9_create_compressor` →
`vp9_encode` → `vp9_encode_tiles_mt` / `_row_mt`).
**HLD anchor:** `notes/hld/01-threading.md`. Every strategy reference in
this document (Strategy A through Strategy F) points back to
HLD-threading §6. The `cpi->td` field catalogue (HLD-threading §5),
the worker-write-surface (HLD-threading §5.9), and the `Send`/`Sync`
table (HLD-threading §7) are normative; this LLD does not re-derive
them, it implements them.

This LLD is the largest threading-related artefact in Stage 2. The
warning in PLAN.md §6 Stage 4a "tile-threading correctness under Rust's
aliasing rules is the hardest unsafe-Rust we'll write" applies in full
here: §3 / §4 / §6 collectively define a dozen unsafe sites, each with
a SAFETY template (HLD-arch §5.4 mandates the six-bullet form). Miri,
loom, and TSan coverage are in §9 against every one.

---

## 1. Purpose

The `rvp9_encoder::threading` module owns the worker pool, per-tile
state, per-worker state, the row-MT job queue with work-stealing, and
the strided slice-split machinery that hands tile-disjoint and
SB-disjoint slices to workers for the duration of one frame encode.

Concretely:

- Persistent OS-thread pool that survives across frames, only respawned
  on `Encoder::reconfigure` (mirroring libvpx `create_enc_workers`
  behaviour at `vp9/encoder/vp9_ethread.c:80-138`).
- One `ThreadData` arena per worker, allocated 32-byte-aligned on
  encoder init, holding `MacroBlock`, `RdCounts`, `FrameCounts`,
  `pc_tree`, `pc_root`, `leaf_tree`, and the `SpeedFeatures` snapshot.
  Mirrors `vp9/encoder/vp9_encoder.h:384-392` (`ThreadData`).
- One `TileSlot` per tile column × tile row, holding `thresh_freq_fact`,
  `mode_map`, `row_mt_sync`, plus the tile-column slices of frame-level
  buffers (above_context, mi_grid, reconstruction planes, denoiser
  averages, segmentation map). Mirrors `vp9/encoder/vp9_encoder.h:348+`
  (`TileDataEnc`).
- A `FrameScope<'frame>` guard that grants workers their per-frame
  borrows, joins them on drop, and bounds every `'static`-cast worker
  closure's actual borrow lifetime. The Rust analogue of libvpx's
  `launch_enc_workers`/`sync()` pattern at
  `vp9/encoder/vp9_ethread.c:140-170`.
- A `RowMtCtx<'frame>` with one job queue per tile column and per-row
  `RowSync` (mutex + condvar + `cur_col` atomic) implementing
  `vp9_row_mt_sync_read` / `_write` from
  `vp9/encoder/vp9_ethread.c:289-413` plus the work-stealing logic from
  `vp9/encoder/vp9_multi_thread.c:309-341`.

The module does **not** own:

- The `BufferPool` (lives in `rvp9-core::frame`; see HLD-arch §3.2).
- `SpeedFeatures` mastering (HLD-arch §3.6 puts `sf` on `Encoder`; this
  LLD reads only via snapshot copies).
- Loop filter row sync (LLD-loopfilter reuses the row-sync primitive
  defined here; see §10).
- Bitstream packing (LLD-bitstream; scalar single-threaded per L-35,
  see §4.10).

---

## 2. libvpx origin

The byte-for-byte source map for this LLD:

| rvp9 concept | libvpx file:line | Stage 1 ref |
|---|---|---|
| `Worker` struct | `vpx_util/vpx_thread.h:387-420` (`VPxWorker`) + `vpx_util/vpx_thread.c:22-90` (`VPxWorkerImpl`) | 06-threading-svc.md §2.6 |
| `WorkerPool` lifecycle | `vp9/encoder/vp9_ethread.c:80-138` (`create_enc_workers`); `vp9/encoder/vp9_ethread.c:92-95` (`vp9_encode_free_mt_data`) | 06-threading-svc.md §3.2 |
| Tile-MT dispatch | `vp9/encoder/vp9_ethread.c:40-59` (`enc_worker_hook`); `vp9/encoder/vp9_ethread.c:194-248` (`vp9_encode_tiles_mt`) | 06-threading-svc.md §3.3 |
| Row-MT dispatch | `vp9/encoder/vp9_ethread.c:586-618` (`enc_row_mt_worker_hook`); `vp9/encoder/vp9_ethread.c:620-690` (`vp9_encode_tiles_row_mt`) | 06-threading-svc.md §3.4 |
| Row sync primitive | `vp9/encoder/vp9_ethread.c:289-413` (`vp9_row_mt_sync_read/_write`) | 06-threading-svc.md §3.6 |
| Job queue | `vp9/encoder/vp9_multi_thread.c:18-52` (`vp9_enc_grp_get_next_job`); `vp9/encoder/vp9_multi_thread.c:262-297` (`vp9_prepare_job_queue`) | 06-threading-svc.md §2.4 |
| Work stealing | `vp9/encoder/vp9_multi_thread.c:309-341` (`vp9_get_tiles_proc_status`) | 06-threading-svc.md §3.4 |
| Worker pool sizing | `vp9/encoder/vp9_ethread.c:84-90` (SVC top-layer tile count); `vp9/encoder/vp9_ethread.c:194-197` (tile-MT); `vp9/encoder/vp9_ethread.c:625` (row-MT) | 06-threading-svc.md §3.2, §7.11 |
| `ThreadData` storage | `vp9/encoder/vp9_encoder.h:384-392` | 06-threading-svc.md §2.5 |
| `EncWorkerData` per-worker dispatch | `vp9/encoder/vp9_ethread.h:25-30` | 06-threading-svc.md §2.1 |
| `MultiThreadHandle` row-MT context | `vp9/encoder/vp9_encoder.h:361-378` | 06-threading-svc.md §2.3 |
| `TileDataEnc` per-tile state | `vp9/encoder/vp9_encoder.h:348+` | 06-threading-svc.md §4.1 |
| Counts accumulation | `vp9/common/vp9_thread_common.c:488` (`vp9_accumulate_frame_counts`); call site `vp9/encoder/vp9_ethread.c:243-247` | 06-threading-svc.md §2.5, §7.4 |
| `MacroBlock` re-wire (non-RD) | `vp9/encoder/vp9_ethread.c:227-232, 668-673` | 06-threading-svc.md §7.2 |
| Tile geometry | `vp9/common/vp9_tile_common.c:18-22` (`get_tile_offset`); `vp9/common/vp9_tile_common.c:15-16` (`MIN/MAX_TILE_WIDTH_B64`) | 06-threading-svc.md §3.5 |
| `setjmp`/`longjmp` (encoder main only) | `vp9/encoder/vp9_encoder.c:2390-2394`; `vp9/encoder/vp9_cx_iface.c:1381-1386` | 01-architecture.md §3.7 |

Limits (`vp9/encoder/vp9_ethread.h:18-20`):

```c
#define MAX_NUM_TILE_COLS (1 << 6)   // 64
#define MAX_NUM_TILE_ROWS 4
#define MAX_NUM_THREADS   80
```

We mirror these as `pub const MAX_NUM_TILE_COLS: u32 = 64;`,
`pub const MAX_NUM_TILE_ROWS: u32 = 4;`, `pub const MAX_NUM_THREADS:
u32 = 80;` in `rvp9_encoder::threading::limits`. Even though
`tile_rows > 1` is rejected in v1 (L-62, §3.5), we retain the constant
to make the v2 expansion path mechanical.

Sync range (`vp9/encoder/vp9_ethread.c:318`):

```c
row_mt_sync->sync_range = 1;
```

Hard-coded for the encoder. `pub const ENCODER_ROW_MT_SYNC_RANGE: i32
= 1;` in `rvp9_encoder::threading::row_sync`. Decoder loopfilter uses a
resolution-dependent variant (`vp9/common/vp9_thread_common.c:265-276`);
LLD-loopfilter owns that.

---

## 3. Public API

The threading module's API is `pub(crate)` to `rvp9-encoder` (HLD-arch
§1.4). Module layout `rvp9-encoder/src/threading/`:

- `worker.rs` — §4.1 `Worker`, `WorkerStatus`, `WorkerJob`
- `pool.rs` — §4.2 `WorkerPool`
- `scope.rs` — §4.3 `FrameScope<'pool, 'frame>`
- `tile.rs` — §4.4 `TileGrid`, `TileSlot<'frame>`,
  `TileColumnArena`, `SbArena`
- `row_sync.rs` — §4.5 `RowSync`
- `job_queue.rs` — §4.6 `RowMtCtx`, `RowMtQueue`, `JobNode`
- `thread_data.rs` — §4.7 `ThreadData<'frame>`, `ThreadDataScratch`
- `limits.rs` — §2 constants

### 3.1 `WorkerPool` API

```rust
impl WorkerPool {
    /// Construct empty pool; does NOT spawn OS threads (lazy per L-31,
    /// mirrors libvpx `create_enc_workers` at
    /// `vp9/encoder/vp9_ethread.c:80-138`). `rtcd_dispatch` enforces
    /// R-09 by construction — caller must have called
    /// `rvp9_core::dsp::dispatch::init()`.
    pub(crate) fn new(rtcd_dispatch: &'static SimdDispatch) -> Self;

    /// Ensure pool has `target` workers (incl. inline slot, §4.2.1).
    /// Tears down and rebuilds on mismatch; mirrors
    /// `vp9_encode_free_mt_data` + `create_enc_workers` at
    /// `vp9_ethread.c:92-95, 103-137`. `target <= MAX_NUM_THREADS`
    /// (debug_assert). Errors: `OutOfMemory`, `SpawnFailed(io::Error)`.
    /// Precondition: no outstanding `FrameScope` (borrow-checked via
    /// `&mut self`).
    pub(crate) fn ensure_sized(&mut self, target: usize)
        -> Result<(), WorkerError>;

    pub(crate) fn num_workers(&self) -> usize;

    /// Open a scoped per-frame borrow region. Returned `FrameScope`
    /// joins all workers on `Drop`. `'pool: 'frame` outlives bound is
    /// load-bearing (workers' `'frame` borrows can't escape the
    /// scope).
    pub(crate) fn open_scope<'pool, 'frame>(&'pool mut self)
        -> FrameScope<'pool, 'frame>
    where 'pool: 'frame;
}

impl Drop for WorkerPool { fn drop(&mut self); }
```

### 3.2 `FrameScope` API

```rust
impl<'pool, 'frame> FrameScope<'pool, 'frame> where 'pool: 'frame {
    /// Tile-MT dispatch. Mirrors `vp9_encode_tiles_mt` at
    /// `vp9_ethread.c:194-248`. Blocks until every worker is `Done`.
    /// On any `Err`, first error returned; all workers still awaited
    /// (no preemption).
    pub(crate) fn run_tile_mt<F>(&mut self,
        ctx: &'frame FrameReadOnly,
        tiles: &'frame [TileSlot<'frame>],
        thread_datas: &'frame mut [ThreadData<'frame>],
        body: F) -> Result<(), WorkerError>
    where F: Fn(&'frame FrameReadOnly,
                &mut ThreadData<'frame>,
                &TileSlot<'frame>,
                u32 /*tile_row*/, u32 /*tile_col*/)
              -> Result<(), WorkerError>
           + Send + Sync + 'frame;

    /// Row-MT dispatch. Mirrors `vp9_encode_tiles_row_mt` at
    /// `vp9_ethread.c:620-690`. Workers pop jobs from `row_mt`;
    /// work-stealing enabled by default, disabled by
    /// `cfg.row_mt_bit_exact || env(RVP9_DETERMINISTIC_ROW_MT=1)`.
    pub(crate) fn run_row_mt<F>(&mut self,
        ctx: &'frame FrameReadOnly,
        tiles: &'frame [TileSlot<'frame>],
        thread_datas: &'frame mut [ThreadData<'frame>],
        row_mt: &'frame RowMtCtx<'frame>,
        encode_sb_row: F) -> Result<(), WorkerError>
    where F: Fn(&'frame FrameReadOnly,
                &mut ThreadData<'frame>,
                &TileSlot<'frame>,
                u32, u32, u32 /*sb_row*/)
              -> Result<(), WorkerError>
           + Send + Sync + 'frame;
}

impl<'pool, 'frame> Drop for FrameScope<'pool, 'frame> { /* joins */ }
```

### 3.3 `TileGrid` and `RowMtCtx` API

```rust
impl TileGrid {
    /// Build per-tile state for the frame; mirrors `vp9_init_tile_data`
    /// at `vp9/encoder/vp9_encodeframe.c:5844-5904`. Preconditions:
    /// `log2_tile_cols ∈ [0, 6]`; `log2_tile_rows == 0` (v1 panic
    /// per L-62 / DV-TT-01).
    pub(crate) fn split_for_frame<'frame>(&'frame mut self,
        cm: &'frame mut CommonFrameMut<'frame>,
        log2_tile_cols: u32,
        log2_tile_rows: u32) -> Vec<TileSlot<'frame>>;
}

impl<'frame> RowMtCtx<'frame> {
    /// Reset per-tile-column queues + row syncs for a new frame.
    /// Mirrors `vp9_prepare_job_queue` + `vp9_assign_tile_to_thread`
    /// at `vp9_multi_thread.c:204-297`.
    pub(crate) fn prepare_jobs(&mut self,
        tile_cols: u32, sb_rows_per_tile: u32, num_workers: usize);
}
```

### 3.4 `ThreadData` and accumulator API

```rust
impl<'frame> ThreadData<'frame> {
    /// Reset per-frame state at worker entry. Value-resets `mb` scalar
    /// fields from `template`; rebinds `plane[j].coeff/qcoeff/
    /// dqcoeff/eobs` to this thread's `pc_root` scratch (the libvpx
    /// `vp9_ethread.c:227-232, 668-673` non-RD re-wire — D-11 / §6.5);
    /// copies `template_sf` into `snapshot_sf` (D-10); memcpy's
    /// `template_counts` into `self.counts` unless inline (L-57);
    /// zeroes `rd_counts`.
    ///
    /// `unsafe fn` per §6.5 — caller (per-worker dispatch closure)
    /// guarantees exclusive access to `self` and `td_scratch`.
    pub(crate) unsafe fn reset_for_frame(&mut self,
        template_mb: &MacroBlock<'frame>,
        template_sf: &SpeedFeatures,
        template_counts: &FrameCounts,
        td_scratch: &'frame mut ThreadDataScratch);

    pub(crate) fn mb(&mut self) -> &mut MacroBlock<'frame>;
    pub(crate) fn counts_mut(&mut self) -> &mut FrameCounts;
    pub(crate) fn rd_counts_mut(&mut self) -> &mut RdCounts;
    pub(crate) fn sf(&self) -> &SpeedFeatures;
}

/// Fold per-worker `FrameCounts` into `dst`. `worker_counts[N-1]` is
/// `None` (inline slot already aliases `dst`). Scalar-only per L-35.
/// Mirrors `vp9_accumulate_frame_counts` at
/// `vp9/common/vp9_thread_common.c:488`; called from
/// `vp9_ethread.c:243-247`. The libvpx baseline-double-count bias
/// is mirrored (DV-TT-05).
pub(crate) fn accumulate_frame_counts(dst: &mut FrameCounts,
    worker_counts: &[Option<&FrameCounts>]);

/// Same shape for `RdCounts`. Mirrors `accumulate_rd_opt` at
/// `vp9_ethread.c:243-247`.
pub(crate) fn accumulate_rd_opt(dst: &mut RdCounts,
    worker_rd: &[Option<&RdCounts>]);
```

### 3.5 Public errors

```rust
#[derive(Debug)]
pub enum WorkerError {
    /// Allocation failed during pool spawn or ThreadData setup.
    /// Pathological — buffers are pre-sized, but malloc can fail.
    OutOfMemory { what: &'static str, bytes: usize },

    /// `std::thread::spawn` returned Err; usually RLIMIT_NPROC or
    /// out-of-memory in the kernel-side stack allocation.
    SpawnFailed(std::io::Error),

    /// A worker hook returned an internal error.
    EncodeFailed {
        worker_id: u32,
        kind: InternalError,
    },

    /// A worker panicked. Under `panic = "abort"` this is
    /// unreachable (process aborts); under `panic = "unwind"`
    /// FrameScope::drop catch_unwinds and converts.
    Panicked {
        worker_id: u32,
        message: String,
    },

    /// Invariant violation, e.g. job queue popped a tile that should
    /// be marked complete. Indicates a logic bug.
    InvariantViolation {
        msg: &'static str,
        location: &'static std::panic::Location<'static>,
    },
}
```

`WorkerError: Send + Sync + 'static` (HLD-arch §4.2 — errors must
cross thread boundaries). `From<WorkerError> for EncodeError` is
defined by the encoder error model (HLD-arch §4).

---

## 4. Types

### 4.1 `Worker`, `WorkerStatus`, `WorkerJob`

```rust
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

pub(crate) struct Worker {
    /// `Some(h)` for OS-thread workers; `None` for the inline slot
    /// (last worker = main thread; mirrors `vp9_ethread.c:131-135`).
    handle: Option<JoinHandle<()>>,
    /// Shared state + condvar. Thread waits on the cond while `Idle`;
    /// main thread signals after installing a job.
    state: Arc<(Mutex<WorkerStatus>, Condvar)>,
    /// Stable id; inline slot is always `id == num_workers - 1`.
    worker_id: u32,
}

/// Worker state machine. Transitions made only under `state.0` lock:
///   Idle -> Running{job}  (main submits)
///   Running -> Done{result} (worker completes)
///   Done -> Idle           (main observes)
///   *    -> Shutdown       (main tears down)
pub(crate) enum WorkerStatus {
    Idle,
    Running { job: WorkerJob },
    Done    { result: Result<(), WorkerError> },
    Shutdown,
}

pub(crate) struct WorkerJob {
    closure: Box<dyn FnOnce() -> Result<(), WorkerError>
                 + Send + 'static>,
    /// `Cell<()>` makes WorkerJob `!Sync` (only one worker runs it).
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

// SAFETY (template applies to every `unsafe impl` in this LLD):
// - Provenance: closure boxed inside `FrameScope::run_*` with
//   captured `&'frame ...` refs.
// - Bounds / Alignment: trivial (Box / trait object).
// - Aliasing: closure moved exclusively into Box; worker drops it
//   after Done; no concurrent access.
// - Lifetime: 'static is a CAST. Real lifetime is 'frame, bounded
//   by `FrameScope::Drop` joining all workers (worker can't outlive
//   scope; closure can't outlive worker).
// - CPU feature: n/a.
unsafe impl Send for WorkerJob {}
```

Stable Rust has no `impl !Sync`; we use `PhantomData<Cell<()>>`
throughout this LLD to suppress auto-Sync (matches HLD-threading
§7.3 footnote).

### 4.2 `WorkerPool`

```rust
pub(crate) struct WorkerPool {
    /// Indexed `0..num_workers`. Last index is the inline slot.
    workers: Vec<Worker>,
    /// One per worker. Holds `pc_tree/pc_root/leaf_tree` and the
    /// 32-byte-aligned `coeff/qcoeff/dqcoeff/eobs_pbuf[j]`. Mirrors
    /// libvpx allocation at `vp9_ethread.c:121` (`vp9_setup_pc_tree`).
    /// MacroBlock plane[j] pointers alias into here (D-11 / §6.5).
    scratch: Vec<Box<ThreadDataScratch>>,
    /// `&'static` from `OnceLock<SimdDispatch>`. R-09 enforced by
    /// type system: cannot construct pool without populated table.
    rtcd_dispatch: &'static SimdDispatch,
    is_shutdown: bool,
    /// !Sync marker.
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}
```

`WorkerPool: Send`, `!Sync` (HLD-threading §7.2). `Encoder` owns the
pool; concurrent access is blocked by `&mut Encoder` exclusivity
during a frame, and by single-threaded between-frame access otherwise.

#### 4.2.1 The inline-slot trick

libvpx (`vp9/encoder/vp9_ethread.c:131-135`):

```c
if (i < num_workers - 1) {
    /* heap-allocate ThreadData */
} else {
    thread_data->td = &cpi->td;   // worker N-1 aliases main thread
}
```

In rvp9 we keep the same invariant: `workers[num_workers - 1]` is the
inline slot. Its `handle: Option<JoinHandle<()>>` is `None`. The
inline worker runs synchronously on the main thread inside
`FrameScope::run_tile_mt` / `run_row_mt` *after* the OS-thread
workers have been launched (so that allocations to `cpi->td` inside
the inline worker don't race with anything).

The accumulator (§3.7, §6.5) skips index `N-1` because:

- Its `counts` storage IS the main thread's `cm.counts`; double-
  accumulating would add the post-encode state to itself.
- Its `rd_counts` storage IS the main thread's `td.rd_counts`; same
  reason.

This is the `vp9/encoder/vp9_ethread.c:243-247` skip behaviour
mirrored exactly. The "skip" is realised in rvp9 by passing
`worker_counts[N-1] = None` to `accumulate_frame_counts`.

#### 4.2.2 Pool sizing

```rust
impl WorkerPool {
    /// Compute the target worker count for a given frame configuration.
    /// Mirrors `vp9/encoder/vp9_ethread.c:84-90, 194-197, 625`.
    pub(crate) fn target_size(
        cfg: &EncoderConfig,
        tile_cols: u32,
        is_row_mt: bool,
        svc_top_layer_tile_cols: Option<u32>,
    ) -> usize {
        let max_threads = cfg.max_threads.min(MAX_NUM_THREADS) as usize;
        if is_row_mt {
            // Row-MT: scale to max_threads, no tile-cols cap
            // (`vp9_ethread.c:625`).
            max_threads.max(1)
        } else if let Some(top_cols) = svc_top_layer_tile_cols {
            // SVC tile-MT: size to the top spatial layer's tile count
            // so we don't re-spawn between layers (L-61;
            // `vp9_ethread.c:84-90`).
            max_threads.min(top_cols as usize).max(1)
        } else {
            // Plain tile-MT: cap at tile_cols
            // (`vp9_ethread.c:194-197`).
            max_threads.min(tile_cols as usize).max(1)
        }
    }
}
```

`cfg.max_threads` is the rvp9 analogue of `cpi->oxcf.max_threads`
(`vp9/encoder/vp9_encoder.h:183`). `cfg.row_mt` is the analogue of
`cpi->oxcf.row_mt`; we set `cpi->row_mt = cfg.row_mt && cfg.max_threads
> 1` at frame start (`vp9/encoder/vp9_encoder.c:6377` writes
`cpi->row_mt = cpi->oxcf.row_mt && oxcf->max_threads > 1`). This is the
**L-59 resolution**: two distinct fields, both passed by config; the
encoder runtime derives the effective row-MT mode per frame.

```rust
// rvp9-encoder/src/config.rs (partial view, owned by HLD-arch / LLD-arch)
pub struct EncoderConfig {
    // ...
    pub max_threads: u32,   // cap on workers
    pub row_mt: bool,       // user preference; combined with max_threads>1
    // ...
}

// At frame start:
let effective_row_mt = cfg.row_mt && cfg.max_threads > 1;
```

The `effective_row_mt` flag is the input to
`WorkerPool::target_size(is_row_mt = effective_row_mt, ...)` and to
the run-tile-MT vs run-row-MT dispatch in `Encoder::encode_frame`.

### 4.3 `FrameScope`

```rust
pub(crate) struct FrameScope<'pool, 'frame> where 'pool: 'frame {
    pool: &'pool mut WorkerPool,
    _frame_lifetime: std::marker::PhantomData<&'frame ()>,
    body_dispatched: bool,
}

impl<'pool, 'frame> Drop for FrameScope<'pool, 'frame> {
    fn drop(&mut self) {
        debug_assert!(self.body_dispatched,
            "FrameScope dropped without dispatching body");
        self.join_all_workers(); // idempotent
    }
}
```

The `'pool: 'frame` bound is **load-bearing**. It proves to the borrow
checker that any `&'frame ...` ref handed to a worker cannot escape
the scope: workers are joined in `Drop`, and `&mut WorkerPool` (held
by `FrameScope`) outlives every worker's borrow. HLD-threading §3.2.

### 4.4 `TileGrid` and `TileSlot`

```rust
// rvp9-encoder/src/threading/tile.rs

/// One per encoder, lifetime-anchored to `Encoder`. Owns the per-tile
/// state (allocated once, re-used across frames) and provides
/// `split_for_frame` to project frame-level buffers into tile slices.
pub(crate) struct TileGrid {
    /// `tile_cols * tile_rows` entries, indexed `row * tile_cols +
    /// col`. Allocated lazily on the first frame; resized only on
    /// `Encoder::reconfigure`.
    tile_storage: Vec<TileStorage>,

    /// Cached geometry from the most recent frame.
    cached_tile_cols: u32,
    cached_tile_rows: u32,
    cached_log2_tile_cols: u32,
    cached_log2_tile_rows: u32,
}

/// Per-tile persistent storage. Survives across frames (the
/// `thresh_freq_fact` table is intentionally retained — it accumulates
/// across the sequence per libvpx
/// `vp9/encoder/vp9_encodeframe.c:5844-5904`).
struct TileStorage {
    thresh_freq_fact: [[i32; MAX_MODES]; BLOCK_SIZES],
    thresh_freq_fact_prev: [[i32; MAX_MODES]; BLOCK_SIZES],
    mode_map: [[i8; MAX_MODES]; BLOCK_SIZES],
    row_base_thresh_freq_fact: Option<Box<[i32]>>,
    row_mt_sync: RowSync,
}

/// One tile's mutable view, lifetime-anchored to `'frame`. Built by
/// `TileGrid::split_for_frame` and consumed by the worker hook.
///
/// SAFETY MODEL:
/// - In tile-MT mode, exactly one worker holds `&mut TileSlot<'frame>`
///   for the frame; aliasing impossible (Send + !Sync).
/// - In row-MT mode, multiple workers hold `&TileSlot<'frame>` and
///   use interior mutability via UnsafeCell on `thresh_freq_fact` and
///   `mode_map`. The happens-before edge is via `RowSync` (§4.5).
///
/// See HLD-threading §6.2 (Strategy B) and §6.3 (Strategy C).
pub(crate) struct TileSlot<'frame> {
    /// Tile geometry (immutable for the frame).
    pub(crate) tile_info: TileInfo,

    /// Tile column index (0..tile_cols).
    pub(crate) tile_col: u32,
    /// Tile row index (0..tile_rows; always 0 in v1, L-62).
    pub(crate) tile_row: u32,

    // ---- per-tile-column slices of frame-level buffers ----
    //
    // These are constructed by `TileGrid::split_for_frame` using
    // `slice::split_at_mut` over the frame buffer. In tile-MT each
    // worker owns its slice exclusively (&'frame mut [...]); in
    // row-MT we wrap them in TileColumnArena (§4.4.3) for interior
    // mutability with bounds checking.
    pub(crate) above_context: TileColumnArena<'frame, EntropyContext>,
    pub(crate) above_seg_context: TileColumnArena<'frame, u8>,
    pub(crate) mi_grid: TileColumnArena<'frame, Option<NonNull<ModeInfo>>>,
    pub(crate) seg_map: TileColumnArena<'frame, u8>,
    pub(crate) y_recon: TileColumnArena<'frame, u8>,
    pub(crate) u_recon: TileColumnArena<'frame, u8>,
    pub(crate) v_recon: TileColumnArena<'frame, u8>,
    pub(crate) denoiser_avg_y: TileColumnArena<'frame, u8>,

    // ---- per-SB scratch (disjoint by SB index) ----
    pub(crate) prev_partition: SbArena<'frame, PartitionContext>,
    pub(crate) prev_segment_id: SbArena<'frame, u8>,
    pub(crate) prev_variance_low: SbArena<'frame, [u8; 25]>,
    pub(crate) content_state_sb_fd: SbArena<'frame, u8>,
    pub(crate) copied_frame_cnt: SbArena<'frame, u8>,
    pub(crate) consec_zero_mv: SbArena<'frame, u8>,

    // ---- per-tile state with interior mutability ----
    pub(crate) thresh_freq_fact: UnsafeCell<&'frame mut [[i32; MAX_MODES]; BLOCK_SIZES]>,
    pub(crate) mode_map: UnsafeCell<&'frame mut [[i8; MAX_MODES]; BLOCK_SIZES]>,
    pub(crate) row_base_thresh_freq_fact: Option<UnsafeCell<&'frame mut [i32]>>,

    // ---- row-MT only ----
    pub(crate) row_sync: &'frame RowSync,
}

// SAFETY: see HLD-threading §7.2.
// - Send: yes (granted by main thread to a worker).
// - Sync: yes ONLY in row-MT mode (multiple workers share &TileSlot);
//         the UnsafeCell fields' aliasing is governed by RowSync.
//         In tile-MT we never share &TileSlot, so Sync is unused.
unsafe impl<'frame> Send for TileSlot<'frame> {}
unsafe impl<'frame> Sync for TileSlot<'frame> {}
```

#### 4.4.1 `TileInfo`

Mirror of libvpx `TileInfo` (`vp9/common/vp9_tile_common.h` and
`vp9/common/vp9_tile_common.c:18-22` for `get_tile_offset`):

```rust
#[derive(Copy, Clone, Debug)]
pub(crate) struct TileInfo {
    /// First MI column (inclusive).
    pub mi_col_start: u32,
    /// Last MI column (exclusive).
    pub mi_col_end: u32,
    /// First MI row (inclusive).
    pub mi_row_start: u32,
    /// Last MI row (exclusive). Always covers the whole frame in v1
    /// (tile_rows == 1).
    pub mi_row_end: u32,
}

impl TileInfo {
    pub fn init(cm: &CommonFrameMetadata, tile_row: u32, tile_col: u32) -> Self {
        // Mirrors vp9_tile_init at vp9_tile_common.c:25-32.
        let mi_col_start = get_tile_offset(tile_col, cm.mi_cols, cm.log2_tile_cols);
        let mi_col_end   = get_tile_offset(tile_col + 1, cm.mi_cols, cm.log2_tile_cols);
        let mi_row_start = get_tile_offset(tile_row, cm.mi_rows, cm.log2_tile_rows);
        let mi_row_end   = get_tile_offset(tile_row + 1, cm.mi_rows, cm.log2_tile_rows);
        Self { mi_col_start, mi_col_end, mi_row_start, mi_row_end }
    }
}

/// SB-64-aligned tile offset. Direct port of
/// `vp9/common/vp9_tile_common.c:18-22`.
fn get_tile_offset(idx: u32, mis: u32, log2: u32) -> u32 {
    const MI_BLOCK_SIZE_LOG2: u32 = 3;
    let sb_cols = mi_cols_aligned_to_sb(mis) >> MI_BLOCK_SIZE_LOG2;
    let offset  = ((idx * sb_cols) >> log2) << MI_BLOCK_SIZE_LOG2;
    offset.min(mis)
}
```

`MIN_TILE_WIDTH_B64 = 4`, `MAX_TILE_WIDTH_B64 = 64`
(`vp9/common/vp9_tile_common.c:15-16`). The legal range for
`log2_tile_cols` follows: for a 1080p frame, `mi_cols = 240`,
`sb_cols = 30`, valid `log2_tile_cols ∈ [0, 5]` such that
`tile_width >= 4 SBs`.

#### 4.4.2 `TileColumnArena` (strided slice wrapper)

Per-tile-column slice of a frame-level strided 2D buffer. In tile-MT
held mutably by one worker; in row-MT shared via `&TileColumnArena`,
intra-column sub-slicing via `unsafe row_slice_mut(sb_row, sb_h)` with
happens-before from `RowSync`. Equivalent libvpx access pattern:
`cm->above_context[mi_col_global..mi_col_global+sb_w]` per
`vp9_ethread.c` and `06-threading-svc.md` §4.1.

```rust
pub(crate) struct TileColumnArena<'frame, T> {
    base: NonNull<T>,
    rows: u32, cols_per_row: u32, stride: u32,
    _marker: PhantomData<&'frame mut [T]>,
}

// SAFETY (HLD-threading §7.2 row table — "TileColumnArena: yes/yes"):
// Outer disjointness from other tile-column arenas guaranteed by the
// `slice::split_at_mut` chain in `TileGrid::split_for_frame` (§4.4.1).
// Inner row-disjointness from other workers in same tile column
// guaranteed by JobQueue invariant + RowSync happens-before.
unsafe impl<'frame, T: Send> Send for TileColumnArena<'frame, T> {}
unsafe impl<'frame, T: Send> Sync for TileColumnArena<'frame, T> {}

impl<'frame, T> TileColumnArena<'frame, T> {
    /// Build from caller's `&'frame mut [T]` after split_at_mut chain.
    /// `rows * stride <= slice.len()` (debug_assert).
    pub(crate) fn from_slice_mut(slice: &'frame mut [T],
        rows: u32, cols_per_row: u32, stride: u32) -> Self;

    /// `&mut [T]` for SB rows `[sb_row * sb_h, (sb_row+1) * sb_h)`.
    ///
    /// SAFETY (caller obligations):
    /// - Provenance: `self.base` from `from_slice_mut`, lifetime 'frame.
    /// - Bounds: `(sb_row + 1) * sb_h <= self.rows` (debug_assert).
    /// - Alignment: T's alignment is preserved (NonNull from slice
    ///   pointer; pointer arithmetic preserves alignment for `T`).
    /// - Aliasing: NO other worker concurrently reads or writes any
    ///   index in `[sb_row*sb_h, (sb_row+1)*sb_h)` within THIS tile
    ///   column. The JobQueue (§4.6) hands each `(sb_row, tile_col)`
    ///   to exactly one worker. For reads by a downstream row, the
    ///   writer of `(sb_row-1, *)` has called `RowSync::write` AND
    ///   this caller has called `RowSync::read`.
    /// - Lifetime: `&'_ mut [T]` is bounded by `&self`, which is
    ///   bounded by `'frame` (PhantomData binding).
    /// - CPU feature: n/a.
    pub(crate) unsafe fn row_slice_mut(&self, sb_row: u32, sb_h: u32)
        -> &mut [T];

    /// Tile-MT exclusive accessor; `&mut self` proves exclusivity, no
    /// SAFETY caller obligation.
    pub(crate) fn full_slice_mut(&mut self) -> &mut [T];
}
```

#### 4.4.3 `SbArena` (per-SB cell array)

For per-SB scratch (e.g. `prev_variance_low`, `consec_zero_mv`).
Disjoint by SB index: JobQueue assigns each `(sb_row, tile_col)` to
exactly one worker, so no two workers write the same SB index.

```rust
pub(crate) struct SbArena<'frame, T> {
    cells: &'frame [UnsafeCell<T>],
}

// SAFETY (HLD-threading §7.2 — "SbArena: yes/yes"):
// One-writer-per-SB invariant supplied by JobQueue.
unsafe impl<'frame, T: Send> Send for SbArena<'frame, T> {}
unsafe impl<'frame, T: Send> Sync for SbArena<'frame, T> {}

impl<'frame, T> SbArena<'frame, T> {
    pub(crate) fn from_cells(cells: &'frame [UnsafeCell<T>]) -> Self;

    /// SAFETY (caller obligations):
    /// - Provenance: `cells` slice, lifetime 'frame.
    /// - Bounds: `sb_index < cells.len()` (debug_assert).
    /// - Alignment: UnsafeCell preserves T's layout.
    /// - Aliasing: no other thread holds `&{ mut }` to
    ///   `cells[sb_index]` during the returned `&mut T`'s borrow.
    ///   JobQueue invariant: one worker per `(sb_row, tile_col)`,
    ///   hence per sb_index.
    /// - Lifetime: returned `&mut T` lives <= `&self`.
    /// - CPU feature: n/a.
    pub(crate) unsafe fn write(&self, sb_index: u32) -> &mut T;

    /// SAFETY (caller obligations):
    /// - Same as `write`, but read-only: caller must guarantee SB at
    ///   sb_index has been written by its owning worker (own writer
    ///   already returned), or by a worker downstream of a released
    ///   RowSync.write edge for the writer's row.
    pub(crate) unsafe fn read(&self, sb_index: u32) -> &T;
}
```

### 4.5 `RowSync`

Mirrors `VP9RowMTSync` at `vp9/encoder/vp9_ethread.h:34-54`. One per
tile column (shared across tile rows in the same column per
`vp9_multi_thread.c:113-119`; in v1 `tile_rows == 1` simplifies to
one RowSync per column).

```rust
pub(crate) struct RowSync {
    /// Per-row Mutex<cur_col[r]>. `cur_col[r]` is the furthest SB
    /// column finished on row r (init -1).
    mutexes: Vec<Mutex<i32>>,
    /// Per-row condvar. Waiter on row r waits on `conds[r-1]`.
    conds: Vec<Condvar>,
    /// Hard-coded `1` for encoder per `vp9_ethread.c:318`. `0` is a
    /// sentinel meaning "dummy / tile-MT mode" (§5.2.2).
    sync_range: i32,
    rows: u32,
    /// Debug-only unsynchronised mirror for instrumentation. Not
    /// load-bearing.
    #[cfg(debug_assertions)]
    cur_col_atomic: Vec<AtomicI32>,
}

// Send + Sync auto-derive from Vec<Mutex<_>> + Vec<Condvar>; no
// manual unsafe impl needed.

impl RowSync {
    pub(crate) fn new(rows: u32) -> Self;
    /// Dummy variant for tile-MT (no-op `read` / `write`); mirrors
    /// `vp9_row_mt_sync_read_dummy` / `_write_dummy` at
    /// `vp9_ethread.c:283-287`. `sync_range == 0` short-circuits.
    pub(crate) fn dummy() -> Self;
    pub(crate) fn reset(&self);

    /// Reader at `(r, c)`. Blocks until `cur_col[r-1] >= c - nsync + 1`.
    /// Mirrors `vp9_row_mt_sync_read` at `vp9_ethread.c:348-366`.
    /// With `nsync == 1` (encoder), predicate simplifies to
    /// `c > cur_col[r-1]`.
    pub(crate) fn read(&self, r: u32, c: i32) {
        let nsync = self.sync_range;
        if nsync == 0 { return; }              // dummy / tile-MT
        if r == 0 || (c & (nsync - 1)) != 0 { return; }
        let prev = (r - 1) as usize;
        let mut g = self.mutexes[prev].lock()
            .expect("RowSync mutex poisoned (worker panic upstream)");
        while c > *g - nsync + 1 {
            g = self.conds[prev].wait(g).expect("cond poisoned");
        }
    }

    /// Writer at `(r, c)`. Updates `cur_col[r]` to `c` (or
    /// `cols + nsync` if last column), signals one waiter. Mirrors
    /// `vp9_row_mt_sync_write` at `vp9_ethread.c:368-413`.
    pub(crate) fn write(&self, r: u32, c: i32, cols: i32) {
        let nsync = self.sync_range;
        if nsync == 0 { return; }
        let (cur, sig);
        if c < cols - 1 {
            cur = c;
            sig = (c % nsync) == nsync - 1;
        } else {
            cur = cols + nsync;
            sig = true;
        }
        if sig {
            let mut g = self.mutexes[r as usize].lock()
                .expect("RowSync mutex poisoned");
            *g = cur;
            #[cfg(debug_assertions)]
            self.cur_col_atomic[r as usize].store(cur, Ordering::Release);
            self.conds[r as usize].notify_one();
        }
    }
}
```

Mutex poisoning policy (HLD-threading §7.4): we `.expect()`. A
poisoned mutex implies an upstream worker panicked; the panic is
converted to `WorkerError::Panicked` by `FrameScope::drop`'s
`catch_unwind`, so the encoder surfaces a clean error rather than
deadlocking. `parking_lot::Mutex` (non-poisoning) is an OT-3 deferral.

### 4.6 `RowMtCtx`, `RowMtQueue`, `JobNode`

Mirrors `MultiThreadHandle` at `vp9/encoder/vp9_encoder.h:361-378`.

```rust
pub(crate) struct RowMtCtx<'frame> {
    /// One queue per tile column. Mirrors `row_mt_info[tile_col]` at
    /// `vp9_encoder.h:361`.
    queues: Vec<RowMtQueue>,
    /// Round-robin: `thread_i_initial_tile_col = i % tile_cols`.
    /// Mirrors `vp9_assign_tile_to_thread` at
    /// `vp9_multi_thread.c:204-207`.
    thread_id_to_tile_id: Vec<u32>,
    /// One RowSync per tile column (shared across tile rows;
    /// `vp9_multi_thread.c:113-119`).
    row_syncs: Vec<RowSync>,
    tile_cols: u32,
    _marker: PhantomData<&'frame ()>,
}

// SAFETY (HLD-threading §7.2 — "RowMtCtx: yes/yes"):
// - inner Mutex<JobQueueInner> linearises concurrent pops.
// - row_syncs are Sync (Vec<Mutex>+Vec<Condvar>).
// - thread_id_to_tile_id is read-only after prepare_jobs.
unsafe impl<'frame> Send for RowMtCtx<'frame> {}
unsafe impl<'frame> Sync for RowMtCtx<'frame> {}

pub(crate) struct RowMtQueue {
    inner: Mutex<JobQueueInner>,
    /// Drained flag. Released-stored by pop() at drain, acquired-loaded
    /// by steal_target() to skip drained columns lock-free.
    completion_status: AtomicBool,
    /// Stealing scan hint; authoritative value lives in `inner`.
    remaining_hint: AtomicUsize,
}

struct JobQueueInner {
    next_row: i32,         // initial 0
    end_row: i32,          // = sb_rows for tile_rows==1
    tile_row: u32,         // always 0 in v1
    num_jobs_acquired: i32, // mirrors `vp9_job_queue.h:42`
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct JobNode {
    pub sb_row: u32,
    pub tile_col: u32,
    pub tile_row: u32,
}

impl RowMtQueue {
    /// Pop next job. Mirrors `vp9_enc_grp_get_next_job` at
    /// `vp9_multi_thread.c:18-52`. On drain, sets completion_status
    /// (Release) so steal_target can skip lock-free.
    pub(crate) fn pop(&self) -> Option<JobNode>;
    pub(crate) fn is_done(&self) -> bool;
    pub(crate) fn remaining(&self) -> usize;
}

impl<'frame> RowMtCtx<'frame> {
    pub(crate) fn new(tile_cols: u32, sb_rows: u32, num_workers: usize)
        -> Self;
    pub(crate) fn prepare_jobs(&mut self, tile_cols: u32,
        sb_rows_per_tile: u32, num_workers: usize);

    pub(crate) fn pop_for_column(&self, tile_col: u32) -> Option<JobNode>;

    /// Find tile column with most remaining work; None if all drained.
    /// Tie-break: lowest tile_col index (matches libvpx). Mirrors
    /// `vp9_get_tiles_proc_status` at `vp9_multi_thread.c:309-341`.
    pub(crate) fn steal_target(&self, current: u32) -> Option<u32>;

    pub(crate) fn initial_tile_for_worker(&self, worker_id: u32) -> u32;
    pub(crate) fn row_sync_for_column(&self, tile_col: u32) -> &RowSync;
}
```

The `pop()` body in full (the only non-trivial implementation):

```rust
impl RowMtQueue {
    pub(crate) fn pop(&self) -> Option<JobNode> {
        let mut g = self.inner.lock().expect("queue mutex poisoned");
        if g.next_row >= g.end_row {
            self.completion_status.store(true, Ordering::Release);
            self.remaining_hint.store(0, Ordering::Release);
            return None;
        }
        let sb_row = g.next_row as u32;
        g.next_row += 1;
        g.num_jobs_acquired += 1;
        let remaining = (g.end_row - g.next_row).max(0) as usize;
        self.remaining_hint.store(remaining, Ordering::Release);
        let tile_row = g.tile_row;
        Some(JobNode { sb_row, tile_col: 0 /*set by caller*/, tile_row })
    }
}
```

#### 4.6.1 Determinism flag (R-08 / L-59)

```rust
/// True when we want bit-exact row-MT. Set when:
/// - `cfg.row_mt_bit_exact` is true (test/diff harness), OR
/// - the env var `RVP9_DETERMINISTIC_ROW_MT=1` is set.
///
/// Effects:
/// - Work stealing is disabled (`steal_target` returns None
///   unconditionally).
/// - `num_workers` is clamped to `min(num_workers, tile_cols)` so
///   each worker stays in its initial column.
/// - `sf.adaptive_rd_thresh_row_mt` is forced to 1 (so
///   `thresh_freq_fact` writes are guarded by per-row
///   `row_base_thresh_freq_fact` mutexes; mirrors libvpx's
///   `vp9_speed_features.c:910-912, 1090-1092`).
///
/// Divergence-ledger entry **DV-T2**: forcing
/// adaptive_rd_thresh_row_mt = 1 costs <0.3% BD-rate but buys
/// bit-exactness.
pub(crate) fn row_mt_deterministic_mode(cfg: &EncoderConfig) -> bool {
    cfg.row_mt_bit_exact || std::env::var_os("RVP9_DETERMINISTIC_ROW_MT")
        .map_or(false, |v| v == "1")
}
```

### 4.7 `ThreadData` and `ThreadDataScratch`

Mirrors `ThreadData` at `vp9/encoder/vp9_encoder.h:384-392`. The
`'frame` lifetime parameterises the `MacroBlock`'s borrowed-slice
fields; `reset_for_frame` (§6.5) rebinds them at frame start.

```rust
pub(crate) struct ThreadData<'frame> {
    /// MacroBlock with `&'frame` slice fields and `NonNull` scratch
    /// pointers. See §6.5 (D-11) for the aliasing model.
    mb: MacroBlock<'frame>,
    /// Accumulated back to `cm.counts` post-frame (§5.8).
    rd_counts: RdCounts,
    /// `None` for inline worker (its storage aliases `cm.counts`;
    /// L-57). `Box` for heap stability across frames.
    counts: Option<Box<FrameCounts>>,
    /// Per-worker snapshot (D-10). `Copy + Clone` — all fields POD.
    snapshot_sf: SpeedFeatures,
    /// Row-MT only. Mirrors `EncWorkerData.tile_completion_status`
    /// at `vp9_ethread.h:29`; read at `vp9_ethread.c:441-442,
    /// 530-531, 606-607`.
    tile_completion_status: [bool; MAX_NUM_TILE_COLS as usize],
    worker_id: u32,
    _frame: PhantomData<&'frame ()>,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// SAFETY (HLD-threading §7.2 — "ThreadData: yes/no"):
// Held exclusively by one worker; never shared. `!Sync` enforced via
// PhantomData<Cell<()>>.
unsafe impl<'frame> Send for ThreadData<'frame> {}
```

**`ThreadDataScratch`** is the worker's persistent scratch (allocated
once at `WorkerPool::ensure_sized`, re-used across frames). The
`MacroBlock.plane[j].{coeff,qcoeff,dqcoeff,eobs}` `NonNull` pointers
alias into here (the libvpx `vp9_ethread.c:227-232` re-wire; D-11).
It has no `'frame` lifetime — lives as long as the worker.

```rust
pub(crate) struct ThreadDataScratch {
    pub pc_tree: Vec<PcTree>,
    pub pc_root: Box<PcTree>,
    pub leaf_tree: Vec<PickModeContext>,
    /// 32-byte aligned per AVX2 transform kernel constraints (§6.4).
    pub coeff_pbuf:   [Box<[TranLow]>; MAX_MB_PLANE],
    pub qcoeff_pbuf:  [Box<[TranLow]>; MAX_MB_PLANE],
    pub dqcoeff_pbuf: [Box<[TranLow]>; MAX_MB_PLANE],
    pub eobs_pbuf:    [Box<[u16]>;     MAX_MB_PLANE],
}
```

`MAX_MB_PLANE = 3` (Y, U, V). Owned by `WorkerPool`; handed to
`ThreadData::reset_for_frame` as `&'frame mut ThreadDataScratch`.

---

## 5. Internal algorithm

### 5.1 Tile-MT encode (entry point)

Mirrors `vp9_encode_tiles_mt` at `vp9/encoder/vp9_ethread.c:194-248`.
Driver in `Encoder::encode_frame_tile_mt`:

```text
1. tile_cols = 1 << log2_tile_cols
   tile_rows = 1 (v1; log2_tile_rows == 0, L-62)
   num_workers = WorkerPool::target_size(cfg, tile_cols,
                     is_row_mt=false, svc_top=svc.top_tile_cols())
   pool.ensure_sized(num_workers)?
2. tiles = tile_grid.split_for_frame(common, log2_tile_cols, 0)
3. scope = pool.open_scope::<'_, 'frame>()
   scope.run_tile_mt(frame_ro, &tiles, thread_datas,
       |fro, td, slot, trow, tcol| encode_tile(fro, td, slot,
                                                trow, tcol))?
4. accumulate_frame_counts(&mut common.counts, &worker_counts_refs)
   accumulate_rd_opt(&mut main_td.rd_counts, &worker_rd_refs)
```

`worker_counts_refs: Vec<Option<&FrameCounts>>` has
`worker_counts_refs[N-1] = None` (inline aliases `common.counts`,
§6.5); `0..N-1` are `Some(&workers[i].td.counts)`.

#### 5.1.1 Worker hook (tile-MT)

Mirrors `enc_worker_hook` at `vp9_ethread.c:40-59`. Strided loop:
worker `i` claims tiles `i, i+N, i+2N, ...`.

```text
fn enc_tile_worker_hook(fro, td, tiles, worker_id, num_workers)
    -> Result<(), WorkerError>:
    td.reset_for_frame(...)
    let total = fro.tile_rows * fro.tile_cols
    let mut t = worker_id
    while t < total:
        let trow = t / fro.tile_cols
        let tcol = t % fro.tile_cols
        encode_tile(fro, td, &tiles[t as usize], trow, tcol)?
        t += num_workers
    Ok(())
```

`encode_tile` is owned by LLD-partition + LLD-mode-decision. Calls
`RowSync::dummy()` `read`/`write` (no-ops) — tile-MT rows are
sequential by control flow within a tile (06-threading-svc §4.1).

### 5.2 Row-MT encode (entry point)

Mirrors `vp9_encode_tiles_row_mt` at `vp9_ethread.c:620-690`. Same
shape as §5.1, with additions:

- `num_workers = WorkerPool::target_size(cfg, tile_cols,
  is_row_mt=true, ...)` (no tile_cols cap; `vp9_ethread.c:625`).
- `row_mt.prepare_jobs(tile_cols, sb_rows, num_workers)` after
  `tile_grid.split_for_frame`.
- Dispatch via `scope.run_row_mt(..., &row_mt, encode_sb_row)`.
- Post-frame accumulation identical to §5.1.

#### 5.2.1 Worker hook (row-MT, with work stealing)

Mirrors `enc_row_mt_worker_hook` at
`vp9/encoder/vp9_ethread.c:586-618`.

```text
fn enc_row_mt_worker_hook(
    frame_ro: &'frame FrameReadOnly,
    td: &mut ThreadData<'frame>,
    tiles: &'frame [TileSlot<'frame>],
    row_mt: &'frame RowMtCtx<'frame>,
    worker_id: u32,
    deterministic: bool,
) -> Result<(), WorkerError> {
    td.reset_for_frame(...);
    let mut cur_tile_id = row_mt.initial_tile_for_worker(worker_id);
    loop {
        match row_mt.pop_for_column(cur_tile_id) {
            Some(node) => {
                encode_sb_row(
                    frame_ro, td, &tiles[node.tile_col as usize],
                    node.tile_row, node.tile_col, node.sb_row,
                )?;
            }
            None => {
                // Current column drained. Mark complete; find another.
                td.tile_completion_status[cur_tile_id as usize] = true;
                if deterministic {
                    // No work stealing in deterministic mode (§4.6.1).
                    return Ok(());
                }
                match row_mt.steal_target(cur_tile_id) {
                    Some(new_col) => { cur_tile_id = new_col; }
                    None => return Ok(()),
                }
            }
        }
    }
}
```

`encode_sb_row` is owned by LLD-partition. The contract is:

1. At entry: `td.snapshot_sf`, `td.mb`, `td.counts`, `td.rd_counts`
   reflect frame-start state (after `reset_for_frame`).
2. Before encoding SB at `(sb_row, sb_col_in_tile)`:
   `row_mt.row_sync_for_column(tile_col).read(sb_row, sb_col_in_tile)`.
3. After encoding the SB:
   `row_mt.row_sync_for_column(tile_col).write(sb_row, sb_col_in_tile,
   sb_cols_in_tile)`.

Note that `encode_sb_row` does **not** know about row-MT vs tile-MT
internally; it always calls `row_sync.read/write`. In tile-MT we pass
a "dummy" `RowSync` whose `read` is a no-op and `write` is a no-op.
The dispatch:

```rust
// Inside encode_sb_row, the partition loop wraps each SB column
// step in:
row_sync.read(sb_row, sb_col_in_tile);
encode_sb(...);
row_sync.write(sb_row, sb_col_in_tile, sb_cols_in_tile);

// where `row_sync` is either `RowSync` (real) for row-MT or
// `DummyRowSync` (no-op) for tile-MT. We can implement this with
// a thin trait or two methods on RowSync that early-return when
// `sync_range == 0` (tile-MT signalling) — see §5.2.2.
```

#### 5.2.2 Dummy row sync

Tile-MT uses `RowSync::dummy()` (sentinel `sync_range == 0`) so
`read` / `write` early-return. Mirrors libvpx
`vp9_row_mt_sync_read_dummy` / `_write_dummy` at
`vp9_ethread.c:283-287`. Rationale: single type with sentinel branch
is cheaper than a trait-object indirect call per SB step (negligible
either way, but no reason to pay it). Two-strategy comparison
(trait-object vs sentinel) collapses to the sentinel since this LLD
owns all call sites.

### 5.3 `WorkerPool::ensure_sized` (lifecycle)

Mirrors `create_enc_workers` at `vp9/encoder/vp9_ethread.c:103-137`.

```text
ensure_sized(target):
    if workers.len() == target: return Ok
    self.shutdown()  // mirror vp9_encode_free_mt_data:92-95
    for i in 0..target:
        state = Arc::new((Mutex<Idle>, Condvar))
        handle = (i < target - 1)
            ? Some(Builder::new()
                   .name("rvp9-enc-worker-{i}")
                   .stack_size(2 MiB)
                   .spawn(|| worker_loop(state.clone()))?)
            : None                // inline slot
        scratch[i] = ThreadDataScratch::new()?  // mirror
                                  // vp9_setup_pc_tree call at :121
        workers.push(Worker { handle, state, worker_id: i })
    Ok
```

Stack size 2 MiB matches libvpx pthread default. Windows
`thread::Builder` defaults to 1 MiB; we explicitly request 2 MiB
to avoid stack-overflow surprises on Windows. OQ-1 confirmed.

Worker thread loop:

```text
worker_loop(state):
    loop:
        g = state.0.lock().expect("poisoned")
        while *g is Idle: g = state.1.wait(g)
        match *g:
            Shutdown: return
            Running{job}: take(job); *g = Idle; drop(g)
                          result = catch_unwind(job.closure())
                          g = state.0.lock()
                          *g = Done { result }
                          state.1.notify_one()
```

`catch_unwind` converts panics to `WorkerError::Panicked` under
`panic = "unwind"`; under `panic = "abort"` the process aborts and
the catch is a no-op (matching libvpx `vpx_internal_error` semantics).

### 5.4 `Worker::launch` / `Worker::sync`

Mirrors `VPxWorkerInterface.launch` / `.sync` at
`vpx_util/vpx_thread.c:32-73`.

```text
launch(job):
    g = state.0.lock()
    debug_assert(*g == Idle, "worker {worker_id} not idle at launch")
    *g = Running { job }
    state.1.notify_one()

sync() -> Result<(), WorkerError>:
    g = state.0.lock()
    while *g != Done: g = state.1.wait(g)
    let prev = take(*g); *g = Idle
    match prev: Done { result } => result
```

Inline slot (`handle == None`) skips `launch`/`sync` and runs the
closure synchronously inside `catch_unwind`.

### 5.5 `FrameScope::run_tile_mt` (dispatch)

```text
run_tile_mt(ctx, tiles, thread_datas, body):
    let N = pool.workers.len()
    self.body_dispatched = true
    let body_arc = Arc::new(body)
    let (inline_td, os_tds) = thread_datas.split_last_mut()

    // Launch OS workers 0..N-1.
    for (i, td) in os_tds.iter_mut().enumerate():
        let closure = move || enc_tile_worker_hook(
            ctx, td, tiles, i as u32, N as u32);
        let boxed: Box<dyn FnOnce()->Result<(), WorkerError>
                       + Send + 'frame> = Box::new(closure);
        let boxed_static = unsafe {
            // SAFETY: 'frame -> 'static trait-object cast.
            // - Provenance: closure captures &'frame {ctx, tiles, td};
            //   all valid for 'frame.
            // - Bounds: refs bounds-checked at use sites.
            // - Alignment: native Rust refs.
            // - Aliasing: each `td` is a disjoint &mut from
            //   split_last_mut().iter_mut(); ctx and tiles are
            //   shared &'frame.
            // - Lifetime: FrameScope::Drop joins all workers before
            //   'frame ends; closure can't outlive Worker, Worker
            //   can't outlive 'frame.
            // - CPU feature: n/a.
            mem::transmute::<_, Box<dyn FnOnce()->Result<(), WorkerError>
                                     + Send + 'static>>(boxed)
        };
        pool.workers[i].launch(WorkerJob { closure: boxed_static })

    // Run inline worker (N-1) synchronously, AFTER OS workers spawned.
    // Ordering: inline writes cm.counts via aliased td.counts; OS
    // workers' counts memcpy already happened in reset_for_frame
    // (§6.5).
    let inline_result = run_inline(body_arc, ctx, inline_td, tiles,
                                   N-1, N)

    let mut first_err = inline_result.err()
    for i in 0..N-1:
        if let Err(e) = pool.workers[i].sync():
            first_err = first_err.or(Some(e))
    first_err.map_or(Ok(()), Err)
```

`run_inline` runs the strided loop with `worker_id = N-1`. `Arc<F>`
is `Send + Sync` since `F: Send + Sync` (per-worker `td` captured by
move into the closure; no mutable shared state).

### 5.6 `FrameScope::run_row_mt` (dispatch)

Same shape as §5.5 with substitutions:
- Worker hook: `enc_row_mt_worker_hook` (§5.2.1).
- Body gains `sb_row` parameter; invoked once per popped job.
- Closure captures `&'frame RowMtCtx<'frame>` plus `deterministic`
  flag (§4.6.1).
- Inline worker runs same hook with `worker_id = N-1`.

### 5.7 Frame counts accumulation

Mirrors `vp9_accumulate_frame_counts` at
`vp9/common/vp9_thread_common.c:488`. Per-field `wrapping_add` over
every `u32` array in `FrameCounts` (y_mode, uv_mode, partition,
coef_counts, ...). Scalar only per L-35. The per-field unrolling is
either macro-generated or a code-gen shim from a field list owned by
LLD-entropy.

**Libvpx bias** (L-57 / HLD-threading §5.6 / **DV-TT-05**): the
"baseline + sum_workers" pattern double-counts the baseline because
each worker started with a memcpy of the baseline. We mirror exactly.

`worker_counts[N-1] == None` (inline slot's storage IS `dst`; it
already wrote during encode). For `num_workers == 1` (only the inline
slot), no fold runs — bit-exact with libvpx.

### 5.8 RD-opt accumulation

Mirrors `accumulate_rd_opt` at `vp9/encoder/vp9_ethread.c:243-247`.
Per-field sum over `RdCounts` (`coef_counts[plane][tx_size][type]
[ref][band][ctx][tok]`, etc.). Detailed per-field accumulation is
owned by LLD-mode-decision; same `worker_rd[N-1] == None` invariant
as §5.7.

```text
fn accumulate_rd_opt(
    dst: &mut RdCounts,
    worker_rd: &[Option<&RdCounts>],
) {
    for w in worker_rd {
        if let Some(rd) = w {
            // Per-field u32 wrapping_add over RdCounts.
    }
}
```

Skips index N-1 for the same reason as `accumulate_frame_counts`.

### 5.9 Tile geometry assert (L-62)

In `TileGrid::split_for_frame`:

```text
fn split_for_frame(...) -> Vec<TileSlot<'frame>> {
    if log2_tile_rows != 0 {
        // Mirrors L-62 / DV-T1.
        panic!(
            "rvp9 v1 limitation: tile_rows > 1 is not supported \
             (log2_tile_rows={}). Real-time mode never uses tile \
             rows; this configuration is deferred to v2. See \
             notes/divergence-ledger.md entry DV-T1.",
            log2_tile_rows
        );
    }
    // ... proceed with tile_rows = 1
}
```

We panic (not return Err) because this is a programming error: the
encoder's public config validation should reject `log2_tile_rows > 0`
at `Encoder::reconfigure` time. The panic here is a defence-in-depth
backstop.

### 5.10 RTCD init invariant (R-09)

```text
impl WorkerPool {
    pub(crate) fn new(rtcd: &'static SimdDispatch) -> Self {
        // R-09: the caller must have obtained `rtcd` via
        // `rvp9_core::dsp::dispatch::init()`, which is OnceLock-
        // backed. This signature enforces the invariant: you cannot
        // construct a WorkerPool without an initialised dispatch
        // table.
        Self {
            workers: Vec::new(),
            scratch: Vec::new(),
            rtcd_dispatch: rtcd,
            is_shutdown: true,
            _not_sync: std::marker::PhantomData,
        }
    }
}
```

This is the entire R-09 enforcement. The worker hook accesses
`rtcd` via `td.rtcd: &'static SimdDispatch`, populated at
`reset_for_frame` time from the pool's `rtcd_dispatch` field.

By the time any worker hook runs, the dispatch table has been
initialised because:

1. `Encoder::new()` calls `rvp9_core::dsp::dispatch::init()` first
   (HLD-simd §2.2).
2. `Encoder::new()` then calls `WorkerPool::new(rtcd)`.
3. The first frame encode triggers `pool.ensure_sized(...)` which
   spawns OS threads.
4. Each OS thread reads `rtcd` only after a `WorkerJob` is launched
   (post `Encoder::new`).

The `OnceLock`'s release-acquire ordering on `get_or_init` guarantees
the worker thread sees a fully-initialised table.

### 5.11 Worker thread priority (L-63)

**Decision: skip in v1.** Matches HLD-threading §10.2 OT-1. libvpx's
Windows shim sets `SetThreadPriority(THREAD_PRIORITY_ABOVE_NORMAL)`
(`vpx_util/vpx_thread.h:90`); Linux/macOS rely on scheduler defaults.
Rust has no portable priority API on stable. For real-time
deployment we recommend the caller set CPU affinity + SCHED_RR /
SCHED_FIFO at the OS level. **DV-TT-06 ledger entry.** A future
`Worker::set_priority(level)` could be added with `#[cfg(windows)]`
`windows-sys` call; out of v1 scope.

---

## 6. SIMD dispatch

### 6.1 Threading itself is not a SIMD surface

Synchronisation primitives (mutex, condvar, atomics) have no SIMD
variants. The threading module ships **no kernels** of its own.

### 6.2 Threading-adjacent kernels (read-only background)

Per `06-threading-svc.md` §6, these kernels are called from worker
hooks but their SIMD lives in `rvp9-core::dsp`:

| Function | SIMD coverage | Used by |
|---|---|---|
| `accumulate_frame_counts` | **scalar only** (L-35; large struct fold, branch-free; AVX2 fold gives ~3% improvement at most — defer to v2) | tile-MT / row-MT post-frame |
| `vp9_loop_filter_alloc` + LF row jobs | calls `vpx_lpf_*` (SSE2/AVX2/NEON) | LLD-loopfilter owns; reuses `RowSync` |
| Down-sampling for SVC | calls `vpx_scaled_*` and `vp9_resize_*` | LLD-svc owns; reuses `WorkerPool` |

### 6.3 RTCD-before-spawn invariant (R-09)

The only SIMD-related concern this LLD owns is the **init-before-spawn**
invariant (§5.10). `WorkerPool::new` requires `&'static SimdDispatch`,
which can only be obtained by calling `dispatch::init()` first. R-09 is
enforced by the type system.

### 6.4 Worker scratch buffer alignment

`ThreadDataScratch.{coeff,qcoeff,dqcoeff}_pbuf[j]` must be **32-byte
aligned** for AVX2 aligned-load kernels. Rust's `Box::new` does not
guarantee >32B alignment; `ThreadDataScratch::new()` allocates via
`AlignedBuf<TranLow, 32>` (defined in `rvp9-core::util::aligned`;
LLD-transforms owns). Mirrors libvpx `DECLARE_ALIGNED(16, ...)` for
`src_diff` and the 32-byte aligned heap allocation in
`vp9_setup_pc_tree`. `eobs_pbuf[j]` is `u16` so 2-byte align suffices.

### 6.5 Aliasing model — `MacroBlock<'frame>` (D-11 implementation)

(Despite the section being inside "SIMD dispatch", this is the
load-bearing aliasing surface that the rest of the LLD references —
it's kept inside §6 because it touches the SIMD scratch buffers
above. See HLD-threading §6.6 for the strategy decision; this
section implements it.)

libvpx's `td->mb` is value-copied per worker
(`vp9_ethread.c:209-212, 649-652`). `MACROBLOCK` contains many
pointers: outer pointers into shared frame buffers (`plane[j].
above_context`, `plane[j].dst`, `plane[j].pre[]`), and inner
pointers to worker-local scratch (`plane[j].{coeff, qcoeff, dqcoeff,
eobs}` rebound to `pc_root.none.*_pbuf[j][0]` at
`vp9_ethread.c:227-232, 668-673` in the non-RD path).

**Decision (HLD-threading §6.6):** outer pointers become `&'frame mut`
borrows into `TileSlot` slices (Strategy A, lifetime-tracked); inner
pointers become `NonNull<TranLow>` / `NonNull<u16>` (Strategy B,
worker-private; safety argued in SAFETY comment).

```rust
pub(crate) struct MacroBlock<'frame> {
    pub plane: [MacroBlockPlane<'frame>; MAX_MB_PLANE],
    pub e_mbd: MacroBlockD<'frame>,
    // ... scalar fields (huge: tens of KB of MV / RD scratch,
    // partition contexts, segment state) — owned in place per worker.
}

pub(crate) struct MacroBlockPlane<'frame> {
    pub src_diff: [i16; MAX_SB_SQUARE],  // owned scalar; aligned 16
    /// Scratch in this worker's ThreadDataScratch. NOT a borrow:
    /// `MacroBlock` and `ThreadDataScratch` co-live for `'frame` in
    /// the same worker; Rust borrow checker rejects a true `&mut`
    /// here because we hand `&mut ThreadDataScratch` elsewhere.
    pub coeff:  NonNull<TranLow>,
    pub qcoeff: NonNull<TranLow>,
    pub eobs:   NonNull<u16>,
    pub src:    Buf2D<'frame>,           // borrow into Source plane
}

pub(crate) struct MacroBlockDPlane<'frame> {
    pub dqcoeff: NonNull<TranLow>,       // ThreadDataScratch scratch
    pub dst:     Buf2DMut<'frame>,       // borrow into TileSlot recon
    pub pre:     [Buf2D<'frame>; 2],     // borrow into frame_refs
    pub above_context: AboveCtxSlice<'frame>, // row-MT-aware (§4.4.2)
    // ...
}
```

`ThreadData::reset_for_frame` is **`unsafe fn`** because it performs
the libvpx `vp9_ethread.c:227-232` re-wire: it stores `NonNull`
pointers from `td_scratch.pc_root.none.{coeff,qcoeff,dqcoeff}_pbuf[j]`
and `eobs_pbuf[j]` into `self.mb.plane[j]` and
`self.mb.e_mbd.plane[j]`. Mandatory SAFETY comment template at every
re-wire site:

```rust
unsafe fn reset_for_frame(
    &mut self,
    template_mb: &MacroBlock<'frame>,
    template_sf: &SpeedFeatures,
    template_counts: &FrameCounts,
    td_scratch: &'frame mut ThreadDataScratch,
) {
    // 1. Scalar fields: memcpy from template_mb.
    self.mb.copy_scalar_fields_from(template_mb);

    // 2. Rebind outer pointers (borrowed slices) — the TileSlot is
    //    granted at call time by the per-worker dispatch closure;
    //    bindings are checked by the Rust borrow system.
    self.mb.plane[*].src = template_mb.plane[*].src.reborrow();
    // ... etc.

    // 3. Rebind inner pointers (NonNull scratch).
    for j in 0..MAX_MB_PLANE {
        // SAFETY:
        // - Provenance: NonNull constructed from pc_root.none.*_pbuf[j][0]
        //   which is a Box<[T]> in `td_scratch`. Pointer is non-null,
        //   well-aligned (32-byte for TranLow per §6.4; 2-byte for u16),
        //   and bounded by the Box's length (MAX_SB_SQUARE × sizeof(T)).
        // - Bounds: pbuf[j] has length MAX_SB_SQUARE; transform kernels
        //   write at most that. Per-kernel bounds-check is the kernel's
        //   responsibility (see LLD-transforms §6).
        // - Alignment: 32-byte for TranLow buffers (AlignedBuf<TranLow, 32>
        //   in ThreadDataScratch::new), 2-byte for u16 (compiler default).
        //   AVX2 unaligned loads (_mm256_loadu_si256) are safe for any
        //   alignment; aligned loads are only used after a debug_assert.
        // - Aliasing: pc_root.none is owned exclusively by this worker
        //   (per-worker ThreadDataScratch ownership, §4.7). The NonNull
        //   pointers we install in plane[j].{coeff, qcoeff, eobs} and
        //   plane[j].dqcoeff alias the same pc_root.none.*_pbuf[j][0].
        //   Mode decision either reads .none context as scratch via
        //   &mut PickModeContext, or via the plane[j].coeff path; never
        //   interleaved (LLD-mode-decision invariant). The borrow
        //   checker cannot see this; Miri verifies (§9.2).
        // - Lifetime: 'frame. `td_scratch: &'frame mut ThreadDataScratch`
        //   is reborrowed for the frame; the NonNull pointers are bound
        //   by `MacroBlock<'frame>`, which cannot outlive
        //   `td_scratch`'s reborrow (PhantomData<&'frame ()> in
        //   MacroBlock binds them).
        // - CPU feature: n/a (pointer storage, no SIMD intrinsic).
        let coeff_ptr  = unsafe { NonNull::new_unchecked(
            td_scratch.pc_root.none.coeff_pbuf[j].as_mut_ptr()) };
        let qcoeff_ptr = unsafe { NonNull::new_unchecked(
            td_scratch.pc_root.none.qcoeff_pbuf[j].as_mut_ptr()) };
        let dqcoeff_ptr= unsafe { NonNull::new_unchecked(
            td_scratch.pc_root.none.dqcoeff_pbuf[j].as_mut_ptr()) };
        let eobs_ptr   = unsafe { NonNull::new_unchecked(
            td_scratch.pc_root.none.eobs_pbuf[j].as_mut_ptr()) };
        self.mb.plane[j].coeff   = coeff_ptr;
        self.mb.plane[j].qcoeff  = qcoeff_ptr;
        self.mb.plane[j].eobs    = eobs_ptr;
        self.mb.e_mbd.plane[j].dqcoeff = dqcoeff_ptr;
    }

    // 4. SpeedFeatures snapshot (D-10).
    self.snapshot_sf = *template_sf;  // POD Copy

    // 5. FrameCounts memcpy (L-57). Inline slot has counts == None;
    //    storage IS cm.counts via the main thread's borrow.
    if let Some(ref mut my_counts) = self.counts {
        **my_counts = template_counts.clone();
    }

    // 6. RdCounts zero.
    self.rd_counts.reset();
}
```

This is the **hardest unsafe in the project**. The §9.2 Miri target
`mb_rewire.rs` validates the aliasing invariant by exercising the
mode-decision inner loop with worker scratch under tracking flags.

The `Buf2D<'frame>` / `Buf2DMut<'frame>` types (used for
`plane[j].{src, dst, pre[]}`) are defined in `rvp9-core::frame` per
HLD-arch §3.2; they wrap a `*const u8` / `*mut u8` plus `(width,
height, stride)` and carry `&'frame [u8]` / `&'frame mut [u8]`
provenance via PhantomData. LLD-frame-buffers owns them. The
`AboveCtxSlice<'frame>` type for `above_context` is the row-MT-aware
accessor backed by `TileColumnArena` (§4.4.2).

---

## 7. Deliberate divergences from libvpx

HLD-threading queued seven entries (HLD-threading §11). Each entry
below is the rvp9 wording for `notes/divergence-ledger.md`.

### DV-TT-01 — `tile_rows > 1` panics in v1 (L-62)

**Where:** `TileGrid::split_for_frame` panics on `log2_tile_rows > 0`.
libvpx counterpart: `MAX_NUM_TILE_ROWS = 4` at `vp9_ethread.h:19`;
row-MT job queue advances `tile_row` at `vp9_multi_thread.c:282-288`.

**What libvpx does:** supports multi-tile-row frames; real-time mode
never sets `log2_tile_rows > 0` (06-threading-svc §7.12; decoder
asserts `log2_tile_rows == 0` at `vp9_decodeframe.c:2343`).

**What rvp9 does:** reject `log2_tile_rows > 0` at config validation;
panic backstop in `split_for_frame`.

**Why:** D1 (real-time) at PLAN.md §2 res/fps never needs tile rows.
Tile rows useful at 4K+; out of v1 scope.

**Byte-parity:** none (matched configs). **Compression:** none.

**Future:** v2 expansion. `RowMtCtx` already carries `tile_row`
per-job; v2 work is propagating through the partition loop and
sharing `RowSync` across tile rows in the same column.

### DV-TT-02 — Determinism flag forces `adaptive_rd_thresh_row_mt = 1`

**Where:** `row_mt_deterministic_mode` (§4.6.1) gates
`sf.adaptive_rd_thresh_row_mt = 1` + disables work stealing. libvpx
counterpart: `vp9_speed_features.c:910-912, 1090-1092` zeros
`adaptive_rd_thresh` when `!adaptive_rd_thresh_row_mt &&
row_mt_bit_exact && max_threads > 1`.

**What libvpx does:** `row_mt_bit_exact` is a debug knob; non-exact
mode races `thresh_freq_fact[][]` for marginal BD-rate.

**What rvp9 does:** when determinism requested
(`cfg.row_mt_bit_exact || RVP9_DETERMINISTIC_ROW_MT=1`), force
`adaptive_rd_thresh_row_mt = 1` AND disable work stealing.

**Why:** Async A7 #11 clarified non-determinism surfaces only at
cpu-used 7/8 with `adaptive_rd_thresh_row_mt = 1`. We codify the
determinism guarantee for the diff harness; production uses libvpx
default.

**Byte-parity:** zero in matched (non-determinism) configs. With
determinism flag: `<0.3%` BD-rate cost (HLD-threading §4.4).
**Compression:** same.

**Future:** revisit at Stage 4a; if cost is acceptable, default on.

### DV-TT-03 — Workers never invoke `vpx_internal_error`

**Where:** workers use `Result<(), WorkerError>`; no setjmp analogue.
libvpx counterpart: encoder workers never call `vpx_internal_error`
in practice (01-architecture.md §3.7).

**What libvpx does:** workers never raise `vpx_internal_error`;
theoretical capability exists.

**What rvp9 does:** all allocation on main thread before worker spawn;
`Result` propagates; `catch_unwind` bounds panics.

**Why:** main-thread allocation is the natural Rust pattern; `Result`
is idiomatic; `catch_unwind` doesn't disturb OS signal semantics.

**Byte-parity:** none on success path. **Compression:** none.

**Future:** none. Stable Rust has no setjmp.

### DV-TT-04 — Pool tears down + rebuilds on `num_workers` change

**Where:** `WorkerPool::ensure_sized` calls `shutdown` then respawns
on `target != current`. libvpx counterpart: `vp9_ethread.c:92-95`
shows conditional rebuild.

**What libvpx does:** rebuild only when `num_workers` differs; same
as us in the same-size case. For shrinkage both libvpx and rvp9
rebuild rather than reuse a subset.

**What rvp9 does:** same as libvpx; on shrink, full rebuild.

**Why:** subset-reuse is more complex (in-flight closures, partial
pool state). Rebuild cost = pthread spawn × N workers = microseconds,
negligible at 33 ms frame budget.

**Byte-parity:** none. **Compression:** none.

**Future:** monitor at Stage 4a under SVC dynamic-resize; revisit if
reconfiguration becomes a hotspot.

### DV-TT-05 — Mirror libvpx's `FRAME_COUNTS` accumulation bias (L-57)

**Where:** `accumulate_frame_counts` (§5.7). libvpx counterpart:
`vp9_accumulate_frame_counts` at `vp9_thread_common.c:488`, called
from `vp9_ethread.c:243-247`.

**What libvpx does:** sums per-worker `td->counts` into `cm->counts`
which already holds the pre-frame baseline (memcpy'd into each
worker). For `N >= 2` OS-workers the baseline is counted N times.

**What rvp9 does:** mirror exactly. Inline (N-1) has no bias (its
storage IS `cm.counts`); OS-workers (0..N-1) accumulate with bias.

**Why:** byte-parity priority for Stage 4a diff harness. Bias is
sub-0.1% BD-rate per HLD-threading §5.6; near-converged probs absorb
the extra `(N-1) × baseline` fractional perturbation.

**Byte-parity:** zero (matches libvpx). **Compression:** identical
to libvpx; vs. unbiased impl, sub-0.1% BD-rate delta.

**Future:** at Stage 5, evaluate switching to unbiased sum.

### DV-TT-06 — Worker thread priority not elevated (L-63)

**Where:** `WorkerPool::ensure_sized` calls `thread::Builder::spawn`
with no priority adjustment. libvpx counterpart: `vpx_thread.h:90`
calls `SetThreadPriority(THREAD_PRIORITY_ABOVE_NORMAL)` on Windows.

**What libvpx does:** Windows-only priority elevation.

**What rvp9 does:** no priority change on any platform.

**Why:** stable Rust has no portable priority API; `windows-sys`
adds a dep for marginal benefit (~2-5% in CPU-saturated scenarios,
less in single-encoder real-time).

**Byte-parity:** none. **Compression:** none (scheduling concern).

**Future:** add platform-specific hook in v2 if profiling shows
starvation.

### DV-TT-07 — No frame-parallel encoding in v1 (Q4 deferred)

**Where:** `Encoder::encode_frame` is single-frame-in-flight; no
`Encoder::enqueue_frame`. libvpx counterpart: `vp9_encode` is
single-frame-in-flight per A6 §1.

**What libvpx does:** encoder is single-frame-in-flight
(06-threading-svc §1; A8 §5 #6). `cm->frame_parallel_decoding_mode`
is a decoder-side hint, not an encoder mode.

**What rvp9 does:** same — single-frame-in-flight. Intra-frame
parallelism via tile-MT + row-MT only.

**Why:** D6 mirrors libvpx; Q4 has no libvpx precedent.

**Byte-parity:** none. **Compression:** none.

**Future:** v2 design when 1080p60 budget squeezes. HLD-threading
§2.3 `LayerScope::Frame` annotations are the v2 expansion checklist.

---

## 8. Open questions

### 8.1 Non-blocking (Stage 3 / 4a may proceed)

- **OQ-1: Stack size.** 2 MiB explicit (§5.3) matches libvpx pthread
  default; avoids Windows `thread::Builder` 1 MiB default. Bump to
  4 MiB if profiling surfaces stack-overflow; Stage 5 watermark test.
- **OQ-2: Cache-padded per-row mutexes.** `crossbeam-utils::CachePadded`
  on `RowSync.mutexes/conds`. ~64 B per row; worth adding. **Default
  yes**, confirm with cachegrind at Stage 3.
- **OQ-3: `parking_lot::Mutex` vs `std::sync::Mutex`.** Faster under
  contention; extra dep. **Default std**; revisit at Stage 5
  (HLD-threading OT-3).
- **OQ-4: NUMA-aware affinity.** Not relevant to v1 single-socket;
  Stage 5 deferral.
- **OQ-5: Inline-worker interleave order.** Run inline strided loop
  fully before syncing OS workers (matches libvpx). Profile at
  Stage 4a.

### 8.2 Blocking (must resolve before Stage 4a)

- **OQ-B-1: Exact `FrameCounts` / `RdCounts` field shapes.** Owned by
  LLD-entropy and LLD-mode-decision. Accumulator correctness depends
  on per-field `u32` `wrapping_add` matching libvpx's `unsigned int +=`
  (HLD-arch §6 integer-width). **Resolution: dependency documented;
  those LLDs transcribe field-by-field.**
- **OQ-B-2: Inline-slot `ThreadDataScratch`.** Inline worker needs
  its own `pc_root.none.coeff_pbuf` scratch — the
  `vp9_ethread.c:227-232` re-wire applies to inline too. **Resolution:
  `WorkerPool.scratch[N-1]` is the inline worker's scratch**, owned by
  the pool just like OS workers; the only diff is
  `workers[N-1].handle = None`.
- **OQ-B-3: Shutdown protocol.** Protocol: lock state mutex, set
  `Shutdown`, notify, drop, `.join()`. Worker loop observes
  `Shutdown` and returns. **Borrow-checker invariant**
  (`&mut WorkerPool`) plus debug_assert that no worker is `Running`
  or `Done` prevents deadlock. §5.3.
- **OQ-B-4: `reset_for_frame` unsafety.** Yes, `unsafe fn` — the
  `MacroBlock` re-wire (§6.5) re-binds `NonNull<TranLow>` fields.
  Caller (per-worker dispatch closure) guarantees exclusive `&mut`
  access; no concurrency issue.

### 8.3 Forwarded to other LLDs

- **OQ-fwd-1:** `encode_tile` (LLD-partition) consumes
  `&TileSlot<'frame>` and calls `tile_slot.above_context.row_slice_mut`.
- **OQ-fwd-2:** Loop-filter row-sync — LLD-loopfilter constructs its
  own `LfSync` (analogous to `VP9LfSync` at
  `vp9_thread_common.h:25-50`) reusing `RowSync`-shaped primitives.
- **OQ-fwd-3:** SVC pool sizing — LLD-svc owns `svc.top_tile_cols()`.
- **OQ-fwd-4:** EXT_RATECTRL callback — main-thread only, no
  threading concern. LLD-rate-control owns.

---

## 9. Test strategy

### 9.1 Test surface

| Layer | Tests | Tooling | PR-blocking? |
|-------|-------|---------|--------------|
| L1: WorkerPool lifecycle | spawn / shutdown / re-spawn on reconfigure | unit tests | yes |
| L2: Worker dispatch | each worker sees correct tile assignment + ThreadData reset | unit tests with mock encode hook | yes |
| L3: Tile-MT aliasing | slice splits disjoint; no two workers write same byte | Miri | yes (per HLD-threading §9.2) |
| L4: Row-MT happens-before | RowSync.read(r,c) blocks until RowSync.write(r-1, c-1) | loom | yes (per HLD-threading §9.3) |
| L5: Counts accumulation | folded counts equal sum-with-bias (libvpx-quirk) | proptest | yes |
| L6: Work-stealing | each (sb_row, tile_col) job consumed by exactly one worker | loom + stress | yes |
| L7: End-to-end determinism | output bit-identical across `max_threads ∈ {1,2,4,8}` under determinism flag | integration test | yes (Stage 4a) |
| L8: SVC pool resize | tile_cols change mid-sequence triggers correct rebuild | unit test | yes |
| L9: Panic safety | worker panic in OS thread converts to `WorkerError::Panicked` | unit test under `panic = unwind` | yes |
| L10: Mutex poisoning | a poisoned RowSync mutex surfaces a clean encoder error | unit test | yes |

### 9.2 Miri targets

```bash
cargo +nightly miri test -p rvp9-encoder --test threading_miri \
    -- --test-threads=1
```

Specific test files in `crates/rvp9-encoder/tests/threading_miri/`:

- `worker_pool_lifecycle.rs` — spawn / shutdown / reconfigure cycle
  for `WorkerPool` with mock encode closures. Verifies no leaked
  threads, no use-after-free on `ThreadDataScratch`.
- `tile_slot_disjoint.rs` — `TileGrid::split_for_frame` produces
  disjoint `TileSlot`s for tile_cols ∈ {1, 2, 4, 8}; concurrent
  writes from spawned threads do not overlap.
- `mb_rewire.rs` — `ThreadData::reset_for_frame` performs the
  `MacroBlock` re-wire; subsequent writes through
  `plane[j].coeff/qcoeff/...` go to the worker's `pc_root` scratch,
  not to any aliased location.

Settings:
`MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-symbolic-alignment-check
 -Zmiri-strict-provenance"` (HLD-threading §9.2 baseline).

Expected slowdown: ~50×. Run on PRs touching `src/threading/*`.

### 9.3 loom targets

`tests/threading_loom/` (`loom = "0.7"`):

- `row_sync_happens_before.rs` — writer at `(r, c)` × reader at
  `(r+1, c)` exhaustively interleaved. Reader never observes
  uncommitted value. Scenarios: 2-thread baseline, 3-thread chain,
  frame-end `cur_col = cols+nsync` signalling.
- `job_queue_disjoint.rs` — two workers popping from one
  `RowMtQueue`; no `(sb_row, tile_col)` popped twice. Covers Mutex
  acquire ordering, `next_row += 1` increment, `remaining_hint.store`
  release.
- `work_stealing_termination.rs` — three workers, two tile columns,
  one steals to the other after drain. Asserts `completion_status`
  visibility.
- `worker_state_machine.rs` — `Idle → Running → Done → Idle` cycle
  exhaustively interleaved. No double-launch, no skipped Done.

PR-blocking on `src/threading/{worker, pool, scope, job_queue,
row_sync}.rs`.

### 9.4 ThreadSanitizer (TSan)

`RUSTFLAGS="-Zsanitizer=thread"` on nightly. Runs against the
Stage 4a integration test (720p tile-MT + row-MT encode of a 5-frame
fixture). Not PR-blocking (slow); runs in the nightly CI tier. Complements
loom's exhaustive-small with TSan's non-exhaustive-real-workload.

### 9.5 Determinism sweep

`tests/determinism_sweep.rs`: encode the same fixture with
`cfg.row_mt_bit_exact = true` and `cfg.max_threads ∈ {1, 2, 4, 8}`;
all four bitstream outputs must be byte-identical. Also runs with
row-MT off (tile-MT only) — that path is fully deterministic without
the bit-exact flag.

### 9.6 Counts-accumulator property test

`tests/counts_accumulator_proptest.rs`: `proptest` with
`baseline + worker_deltas[0..N]`. Computes expected:

```
cm.counts == N * baseline + sum_i(deltas[i])
```

(the libvpx bias from §5.7 / DV-TT-05). Asserts against the fold
output. Verifies inline-worker direct-write semantics for `N == 1`
(no bias).

### 9.7 SVC pool-resize test

`tests/svc_pool_resize.rs`: encode a 2-layer spatial SVC stream
(360p + 1080p). Verify (a) pool sized to top-layer tile count
(`min(max_threads, 8)`); (b) layer-0 frames launch only 2 workers
(others stay `Idle`); (c) layer-1 frames launch all 8; (d)
`reconfigure` to single-layer 720p (4 tile max) tears down extras.

### 9.8 Panic-safety test

`tests/worker_panic.rs`: dispatch a closure that panics in worker 0
under `cfg(panic = "unwind")`. Assert returned error matches
`WorkerError::Panicked { worker_id: 0, .. }`. Gated by
`#[cfg(panic = "unwind")]` since `panic = "abort"` aborts the
process.

### 9.9 What we do NOT test

- **Performance under contention** — perf tests live in
  `bench/realtime.rs` (HLD-threading §9.7), Stage 4a target.
- **Loopfilter sync** — LLD-loopfilter owns; this LLD provides
  primitives.
- **SVC inter-layer interaction beyond pool sizing** — LLD-svc owns.
- **EXT_RATECTRL callback paths** — LLD-rate-control owns; threading
  is uninvolved (callback runs on main thread).

---

## 10. Cross-cutting downstream consumers

- **LLD-partition** — `encode_tile` consumes `&TileSlot<'frame>`,
  calls `row_sync.read/write` per SB.
- **LLD-mode-decision** — consumes `&mut ThreadData<'frame>`; owns
  the inner MacroBlock encode work.
- **LLD-rate-control** — reads `&Common.counts` post-accumulate.
- **LLD-loopfilter** — reuses `RowSync` for `VP9LfSync` analogue with
  resolution-dependent `sync_range` (`vp9_thread_common.c:265-276`).
- **LLD-denoiser** — `&mut TileSlot.denoiser_avg_y` via
  `TileColumnArena.row_slice_mut`; tile-disjoint.
- **LLD-aq** — `&mut TileSlot.seg_map` for cyclic-refresh
  segment_id; per-block-disjoint.
- **LLD-svc** — provides `svc.top_tile_cols()`; calls
  `WorkerPool::ensure_sized` on layer changes.
- **HLD-simd / LLD-rtcd** — consumes R-09 invariant
  (`WorkerPool::new` requires `&'static SimdDispatch`).
- **HLD-arch** — H1-H8 (HLD-threading §10.1) all satisfied:
  - H1 ✓ (HLD-arch §3.6: `Encoder` owns subsystems).
  - H2 — owned by LLD-entropy + LLD-mode-decision.
  - H3 ✓ (HLD-arch §3.6: `sf` is one `Copy + Clone` struct).
  - H4 ✓ (HLD-arch §4.2: `EncodeError` top-level enum;
    `WorkerError` is internal and wrapped).
  - H5 ✓ (HLD-arch §4.1: `Encoder::new` returns
    `Result<Encoder, EncodeError>`).
  - H6 ✓ (HLD-arch frame-loop policy).
  - H7 ✓ (HLD-arch §3.6: `ThreadData` in `rvp9-encoder::threading`).
  - H8 ✓ (HLD-arch §3.2: `FramePool` in `rvp9-core::frame`).

---

## 11. Summary

Types: `Worker`, `WorkerStatus`, `WorkerJob`; `WorkerPool`;
`FrameScope<'pool, 'frame>`; `TileGrid`, `TileSlot<'frame>`,
`TileColumnArena<'frame, T>`, `SbArena<'frame, T>`; `RowSync`
(encoder hard-codes `sync_range = 1`); `RowMtCtx<'frame>`,
`RowMtQueue`, `JobNode`; `ThreadData<'frame>`, `ThreadDataScratch`.

Key invariants:

1. **R-09:** `WorkerPool::new(&'static SimdDispatch)` enforces RTCD
   init before worker spawn — type-system guarantee.
2. **L-57:** per-worker `Box<FrameCounts>` memcpy'd from `cm.counts`
   at frame start; accumulated back via `accumulate_frame_counts`
   (scalar-only per L-35), skipping inline slot. libvpx
   baseline-double-count bias mirrored (DV-TT-05).
3. **L-59:** `EncoderConfig.row_mt` and `.max_threads` are separate;
   `effective_row_mt = row_mt && max_threads > 1` derived per frame.
4. **L-62:** `log2_tile_rows > 0` panics in v1 (DV-TT-01).
5. **L-63:** worker priority not elevated (DV-TT-06).
6. **L-35:** bitstream packing scalar-only (not driven by this LLD;
   flagged in §6.2, §7).
7. **D-10:** `SpeedFeatures` per-worker snapshot, no aliasing
   (HLD-threading §5.7).
8. **D-11:** `MacroBlock<'frame>` lifetime-parameterised; outer
   `&'frame mut` borrows of `TileSlot`, inner `NonNull` scratch
   pointers into `ThreadDataScratch.pc_root` (§6.5).

`unsafe` sites with SAFETY templates (HLD-arch §5.4 six-bullet form):

- `WorkerJob`: `'frame -> 'static` cast bounded by `FrameScope::Drop`
  (§4.1).
- `TileColumnArena::row_slice_mut`: row-MT shared access; happens-
  before via `RowSync` (§4.4.2).
- `SbArena::{write, read}`: per-SB disjointness via JobQueue
  (§4.4.3).
- `MacroBlock::reset_for_frame`: pointer re-wire to
  `ThreadDataScratch.pc_root` (§6.5).
- Boxed-closure transmute inside `FrameScope::run_tile_mt` /
  `run_row_mt`: trait-object lifetime cast (§5.5).

Test coverage: Miri (§9.2), loom (§9.3), TSan (§9.4), determinism
sweep (§9.5), proptest accumulators (§9.6), SVC resize (§9.7),
panic safety (§9.8). PR-blocking gates per §9.1.

---

**Line count:** ~1990. **Key §7 entries:** DV-TT-01 (`tile_rows > 1`
panic, L-62), DV-TT-02 (determinism flag forces
`adaptive_rd_thresh_row_mt = 1` + disables work stealing, R-08),
DV-TT-03 (no `vpx_internal_error` analog), DV-TT-04 (pool full
rebuild on `num_workers` change), DV-TT-05 (mirror libvpx
`FRAME_COUNTS` baseline-double-count bias, L-57), DV-TT-06 (no
worker priority elevation, L-63), DV-TT-07 (no frame-parallel encode
in v1, Q4). **Blocking §8 questions:** OQ-B-1 forwarded to
LLD-entropy + LLD-mode-decision (FrameCounts/RdCounts exact field
shapes); OQ-B-2/B-3/B-4 resolved in-LLD.
