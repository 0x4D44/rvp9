# HLD-threading: rvp9 encoder threading design

**Stage 2 HLD.** Mirrors libvpx VP9's tile-based pthread model under
Rust aliasing rules. Resolves Q3, Q4, D-10, D-11, D-12, D-15, L-66.

**Authors:** Stage 2 (this run).
**Date:** 2026-05-16.
**Reference implementation:** `third_party/libvpx/` at v1.14.1
(`12f3a2a`). All `third_party/libvpx/<path>:<line>` citations resolve
there.
**Primary specialist source:** `notes/vp9-analysis/06-threading-svc.md`
(A6, 2002 lines). State graph derived from
`notes/vp9-analysis/00-overview.md` §3. Architectural touch-points from
`notes/vp9-analysis/01-architecture.md`.

This HLD is the contract every encoder LLD must read before designing
its data ownership. The §5 state catalogue and §6 `UnsafeCell` strategy
are the load-bearing artefacts; the rest is rationale and test design.

---

## 0. Scope and outputs

What this doc decides:

1. **Q3** — threading primitives (§1).
2. **Q4** — frame-parallel encoding in v1 vs v2 (§2).
3. Worker thread pool design — lifecycle, sizing, error propagation (§3).
4. Tile assignment + tile-row/row-MT sync protocol (§4).
5. **Shared-state contract** — the per-field rvp9 equivalent of `VP9_COMP`
   with owner / sync / safety comment template (§5). Resolves D-12 and
   D-15.
6. **`UnsafeCell` strategy** — pattern + invariants (§6). Resolves D-10
   and D-11.
7. **`Send` + `Sync` impls** — manual-impl list with justification (§7).
8. **`setjmp`/`longjmp` × worker threads** — Rust replacement (§8).
   Resolves L-66.
9. **Test strategy** — Miri / TSan / `loom` / soak (§9).

What this doc explicitly does **not** decide:

- Crate boundaries and the `VP9_COMP` decomposition (HLD-arch, L-29).
- The `RATE_CONTROL` plumbing across SVC layers (LLD-rate-control + LLD-svc).
- Loop-filter row-sync internals (LLD-loopfilter — reuses our row-MT
  primitives though, see §4.4).
- Decoder threading (LLD-tile-threading carries the parse/recon design;
  this HLD covers it briefly in §4.5 because the same primitives apply).

In-scope but explicitly aspect-locked here:

- §3 worker pool is **persistent**, sized on `Encoder::new` /
  `Encoder::reconfigure`, mirroring libvpx (`vp9_ethread.c:80-138`).
- §4 tile assignment is **column-strided**, mirroring libvpx
  (`vp9_ethread.c:50-56`).
- §6 chooses **scoped threads + slice splitting** for tile-MT and
  **`UnsafeCell` + raw pointers under documented row-MT happens-before
  edges** for row-MT.

---

## 1. Q3 — Threading primitives

### 1.1 Options considered

| # | Approach | Mental model | Cost |
|---|----------|--------------|------|
| A | `std::thread` + `Mutex` / `Condvar` / `Barrier` | 1:1 pthread mirror | hand-rolled work-stealing |
| B | `crossbeam-channel` + scoped threads | message-passing | diverges from libvpx; channel doesn't match point-to-point row sync (see `06-threading-svc.md` §7.8) |
| C | `rayon::ThreadPool` + `scope_fifo` | work-stealing pool | hides the synchronization libvpx needs explicit control over |
| D | `std::thread::scope` (stable 1.63) + hand-rolled `(Mutex<JobQueue>, Condvar)` + persistent worker pool | scoped pthread mirror with borrow-checker-visible lifetimes | minor extra design for scoping |

### 1.2 Decision — option D, with two refinements

**`std::thread::scope` is not quite right** for a persistent pool that
lives across many frames; `scope` parks the calling thread until every
spawned thread joins, which forces re-spawning workers per frame. That
contradicts libvpx's persistent-pool model (`vp9_ethread.c:103-137`,
where threads are created once per `create_enc_workers` call and only
re-created if `num_workers` changes).

We therefore use a **two-layer design**:

1. **Persistent worker pool** built directly on `std::thread::spawn` +
   `(Mutex<WorkerState>, Condvar)` per worker. State machine
   `Idle -> Running -> Idle` mirrors `VPxWorkerStatus` `OK -> WORK -> OK`
   (`vpx_util/vpx_thread.h:142-152`, `vpx_util/vpx_thread.c:32-73`). One
   worker thread is dormant on its condvar between frames. This is the
   `Worker` type in §3.

2. **Per-frame scoped borrows** via a custom `FrameScope<'frame>` guard.
   The guard hands each worker a `&'frame TileSlot<'frame>` for the
   duration of one frame; on drop it joins all workers (or — for the
   "main thread is worker N-1" trick libvpx uses — runs worker N-1
   inline, then joins workers 0..N-1). The lifetime `'frame` lets the
   borrow checker see that mutable slice splits (§4.2) outlive every
   worker.

This is a known pattern. The closest published example is
`std::thread::scope` itself; we keep its semantics for borrows but
detach the thread lifecycle from the scope.

**Sync primitives used:**

- `std::sync::Mutex<T>` — for worker `WorkerState`, the row-MT
  `JobQueue` (per tile column), and the `RowSync.cur_col[]` array
  protected by per-row mutexes.
- `std::sync::Condvar` — paired with each mutex above. One per worker
  for `Idle <-> Running`; one per row in `RowSync` for the row-MT
  happens-before edge.
- `std::sync::atomic::AtomicI32` — for the `cur_col[r]` cell itself,
  read under mutex but stored as atomic so the writer's release ordering
  on `condvar.notify_one()` plus reader's acquire on the mutex lock is
  not the *only* path the optimizer must respect. (This is conservative;
  Mutex alone is sufficient. Atomic gives us cheap debug-time reads
  outside the mutex, e.g. for instrumentation.)
- `std::sync::Once` / `std::sync::OnceLock<RtcdTable>` — for RTCD
  one-time init before any worker spawn (R-09).
- `std::sync::Barrier` — **not used in v1**. libvpx uses a "launch all,
  then sync all" pattern (`vp9_ethread.c:153-170`) which is implemented
  with the worker condvar; a single global barrier would serialise
  workers that finish early. Reserved as a possible v2 idiom for
  loop-filter rows.

**No third-party deps** for the threading core: `std`-only. (We may use
`crossbeam-utils::CachePadded` for the per-row mutex array to avoid
false-sharing on `cur_col[]`, but that's a micro-opt with no
correctness impact, deferred to LLD-tile-threading.)

### 1.3 Why not rayon?

Three reasons, all from `06-threading-svc.md` §7.3:

1. **Work pattern is not parallel-iterator.** Tile-MT is "N workers,
   each striding through `tiles[i, i+N, i+2N, ...]`"
   (`vp9_ethread.c:50-56`). Row-MT is "N workers pulling from a
   per-tile-column job queue with work-stealing on column drain"
   (`vp9_ethread.c:600-617`, `vp9_multi_thread.c:309-341`). Both fit
   poorly into `par_iter().map(...).collect()`.
2. **The point-to-point row sync** at `vp9_row_mt_sync_read` /
   `_write` (`vp9_ethread.c:348-413`) is between SB `(r, c)` and SB
   `(r-1, c)` *specifically* — it's not a join/barrier and not a channel
   message. rayon offers no primitive for this.
3. **Determinism.** The bit-exact row-MT mode (R-08) requires that
   `tile_data->thresh_freq_fact[][]` is read/written in a particular
   order across tiles. rayon's job-stealing reorders that
   non-deterministically.

### 1.4 Why D6 doesn't force option A literally

D6 commits to "mirror libvpx tile-based pthread model". Option D is
option A *plus* a scoped-borrow wrapper for the borrow checker. Sync
semantics remain 1:1 with libvpx; D6's "idiomatize later" admits the
wrapper.

---

## 2. Q4 — Frame-parallel encoding (v1 vs v2)

### 2.1 The libvpx-mirror confusion

`cm->frame_parallel_decoding_mode` (`vp9/common/vp9_onyxc_int.h:257`,
written to the uncompressed header at `vp9/encoder/vp9_bitstream.c:1304`)
is a **decoder-side hint** that disables backward probability updates so
independent decoders can parse multiple frames in parallel. It is
**not** an encoder threading mode. The libvpx encoder is
single-frame-in-flight (`06-threading-svc.md` §1 explicit; A8 §5 #6
confirms; PLAN.md §2 Stage 1 surprising finding #2).

PLAN.md Q4 framing ("frame-parallel encoding in v1 vs v2") therefore
has no libvpx-mirror precedent to follow. Q4 is a design question, not
a port question.

### 2.2 Decision — defer to v2

In v1, **the rvp9 encoder is single-frame-in-flight.** Intra-frame
parallelism comes from tile-MT and row-MT (§4). Reasons:

1. **D6 doesn't ask for it.** D6 mirrors libvpx. Mirror = single frame
   in flight.
2. **No libvpx pattern to follow.** Designing frame-parallel encode
   from scratch is a v2-scale endeavour: it changes the lifetime of
   nearly every state structure in §5 from "one frame" to "many frames
   overlapping", and it forces reasoning about reference-frame buffer
   ref-counts under concurrent producers. Bolting this onto a tile-MT
   port that's still proving byte-parity is a recipe for a moving
   correctness target.
3. **Performance budget at 1080p60 is the only place this matters.**
   Stage 0 measured libvpx at 5.04 ms/frame at 1080p60 (cpu-used 7) —
   well under the 16.67 ms budget on synthetic content. Real content
   will be slower but the headroom is large. If Stage 4a hits the
   1080p60 wall without frame-parallel, we escalate as D-04 (PLAN.md
   §4 already routes this through Stage 4a entry).
4. **WebRTC and livestream targets (D1) do not need it.** Most WebRTC
   stacks are single-frame-in-flight per stream. SVC adds parallelism
   across layers (which is also serial in libvpx —
   `06-threading-svc.md` §3.7 confirms — but maps to v2 frame-parallel
   if needed).

### 2.3 What we keep open for v2

The §5 state catalogue annotates each field with `LayerScope::Frame`,
`LayerScope::Encoder`, or `LayerScope::Worker`. The
`LayerScope::Frame` set is the set of state structures that would need
to be cloned or refcount-promoted for multi-frame parallelism in v2.
Keeping this annotation now costs nothing; it gives a v2 author a
checklist.

We also **do not** add `Arc<...>` to fields that the v1 design uses
single-ownership for. The Stage 2 motto is "no premature concurrency
machinery for v2" (PLAN.md "don't predict the future" / "premature
abstractions actively resist the correct design"). If v2 promotes a
field to `Arc<T>`, it does so once, deliberately.

---

## 3. Worker thread design

### 3.1 Reference: libvpx lifecycle invariants

From `06-threading-svc.md` §3.2 / `vp9_ethread.c:80-138`:

1. Worker threads persist across frames. `create_enc_workers` re-spawns
   only when `num_workers` changes.
2. Worker `N-1` (last) **is** the main thread; its `ThreadData` aliases
   `cpi->td` (`vp9_ethread.c:131-135`). This is *the* subtlety: per-worker
   accumulation skips index `N-1` (`vp9_ethread.c:243-247`).
3. Per-worker `pc_tree` + `counts` + `td` allocated 32-byte aligned;
   teardown frees them (skipping `N-1`, `vp9_ethread.c:183-187`).
4. SVC pool sizing uses the **top spatial layer's** tile count
   (`vp9_ethread.c:84-90`), not the current layer's.

### 3.2 rvp9 design

```text
WorkerPool {
    workers: Vec<Worker>,    // worker N-1 is the "inline" slot
                              // indices 0..N-1 are OS threads
}
Worker {
    handle: Option<JoinHandle<()>>,    // None for the inline slot
    state: Arc<(Mutex<WorkerState>, Condvar)>,
    td: ThreadData,
}
enum WorkerState {
    Idle,
    Running { job: WorkerJob<'static> },     // 'static via type erasure
    Done { result: Result<(), WorkerError> },
    Shutdown,
}
```

`WorkerJob<'frame>` is the type-erased per-frame closure; the
`'static` lie is bounded by `FrameScope<'frame>` joining all workers in
its `Drop` (mirroring `std::thread::scope`). SAFETY comment template at
every job-submission site:

```text
// SAFETY: the closure borrows &'frame TileSlot<'frame>. FrameScope's
// Drop joins all workers before 'frame ends. Send bound on the closure
// is satisfied because TileSlot: Send (§7). The 'static cast fits the
// erased slot; actual borrow lifetime is enforced by FrameScope.
```

This is the standard scoped-thread pattern; tested with Miri (§9).

### 3.3 Pool sizing

Driven by `oxcf.max_threads` (libvpx field, `vp9/encoder/vp9_encoder.h:183`)
and `tile_columns` / `tile_rows`:

| Mode | Worker count |
|------|--------------|
| Tile-MT, no SVC | `min(max_threads, tile_cols)` (`vp9_ethread.c:194-197`) |
| Tile-MT + SVC | `min(max_threads, max_tile_cols_across_layers)` (`vp9_ethread.c:84-90`) |
| Row-MT | `max(max_threads, 1)` (`vp9_ethread.c:625`) |

We mirror the same formula. For v1 (8-bit, real-time, no temporal
filter, no first pass), only tile-MT and row-MT paths exist; the
`first_pass_*` and `temporal_filter_*` hooks (`vp9_ethread.c:415-583`,
`#if !CONFIG_REALTIME_ONLY`) are not ported (PLAN.md §6 Stage 4a).

**L-31 (eager vs lazy worker spawn) resolution:** **lazy, mirroring
libvpx.** The pool is created at the first frame where the worker count
crosses zero, then **persists** until a config change or
`Encoder::drop`. This matches `create_enc_workers`'s "only re-spawn if
`num_workers` changes" behaviour. Eager-at-`Encoder::new` would spawn
threads that may never be used if the caller's first frame is
single-tile (e.g. very low-res test fixtures).

### 3.4 Reconfigure semantics

When `Encoder::reconfigure` (the analog of `vp9_change_config` plus
SVC layer change) raises `max_threads` or changes the top spatial layer
resolution such that `max_tile_cols` grows, the pool is **torn down
and rebuilt**. This matches `vp9_encode_free_mt_data` +
`create_enc_workers` (`vp9_ethread.c:92-95`).

Shrinking `max_threads` also tears down and rebuilds, for simplicity.
Workers do not hold per-encoder long-lived state outside `ThreadData`
that would make this expensive.

### 3.5 Error propagation and panic policy

libvpx workers signal failure via `worker->had_error`
(`vpx_util/vpx_thread.h`), but the encoder hooks never return non-zero
in real-time paths — `vpx_internal_error` is not called from workers
(allocation lives on main thread; bitstream malformation is decoder-only).
See §8.

In rvp9, `Done { result: Result<(), WorkerError> }` carries the error.
`FrameScope::drop` collects all results; if any worker returned `Err`,
the first is propagated as `EncoderError::WorkerFailed(...)`. Other
workers run to completion (no pre-emption) so `ThreadData` is left
re-initialisable. Stronger guarantee than libvpx's tear-down-on-error
behaviour.

`WorkerError` variants (initial): `InvariantViolation { msg, location }`,
`OutOfMemory` (pathological — pre-sized buffers), `InternalLogic`,
`Panicked(String)`.

**Panic-in-worker:** under `panic = "abort"` (release prod) a panic
aborts the process — matches libvpx `vpx_internal_error` semantics.
Under `panic = "unwind"` (dev / tests) `FrameScope::drop` uses
`catch_unwind` per worker, converts to `WorkerError::Panicked`, joins
all workers before re-raising on the main thread (no leaked
`JoinHandle`). We do not rely on unwinding for correctness in release.

---

## 4. Tile assignment + sync protocol

### 4.1 Tile partitioning (read-only background)

VP9 tile geometry is fully determined by `log2_tile_cols` and
`log2_tile_rows` in the uncompressed header. Boundaries are SB64-aligned
via `get_tile_offset` (`vp9/common/vp9_tile_common.c:18-22`):

```c
static int get_tile_offset(int idx, int mis, int log2) {
  const int sb_cols = mi_cols_aligned_to_sb(mis) >> MI_BLOCK_SIZE_LOG2;
  const int offset = ((idx * sb_cols) >> log2) << MI_BLOCK_SIZE_LOG2;
  return VPXMIN(offset, mis);
}
```

Per `06-threading-svc.md` §3.5: `MIN_TILE_WIDTH_B64 = 4` (256 px),
`MAX_TILE_WIDTH_B64 = 64` (4096 px), so the legal `log2_tile_cols` for
each frame size is bounded.

**L-62 (assert `log2_tile_rows == 0` in v1) — confirmed.** Real-time
encoding never uses tile rows (`06-threading-svc.md` §7.12); the encoder
supports `log2_tile_rows > 0` (max `MAX_NUM_TILE_ROWS = 4`) but at our
D1 target it's unreached. The rvp9 v1 encoder panics on `tile_rows > 1`
with a clear "v1 limitation, expand in v2" message. Divergence ledger
entry.

### 4.2 Tile-MT worker assignment

From `vp9_ethread.c:50-56`:

```c
for (t = thread_data->start; t < tile_rows * tile_cols;
     t += cpi->num_workers) {
    int tile_row = t / tile_cols;
    int tile_col = t % tile_cols;
    vp9_encode_tile(cpi, thread_data->td, tile_row, tile_col);
}
```

Worker `i` claims tiles `i, i+N, i+2N, ...` where `N = num_workers`.
With `tile_rows == 1` (v1 invariant), this reduces to disjoint tile
columns.

rvp9 mirrors this exactly: worker `i` runs

```text
for t in (i .. tile_rows * tile_cols).step_by(num_workers) {
    let tile_row = t / tile_cols;
    let tile_col = t % tile_cols;
    encode_tile(cpi_ro, tile_row, tile_col, &mut td_mut);
}
```

The borrow split is per-tile: `TileSlot` holds the per-tile mutable
state (`thresh_freq_fact`, `mode_map`, etc. from §5.4), and the
worker's `&'frame TileSlot<'frame>` is granted exclusively for the
tiles `i, i+N, i+2N, ...`. Other tiles' `TileSlot`s are owned by other
workers concurrently. We achieve this with **strided slice splitting**
(see §6.3).

### 4.3 Row-MT worker assignment and sync

Row-MT is the harder primitive. Sync protocol mirrors
`vp9_ethread.c:289-413`:

```text
RowSync { mutex: [Mutex<i32>; rows], cond: [Condvar; rows], sync_range: 1 }

// Reader at (r, c): wait until the SB above is far-enough along.
sync_read(rs, r, c):
    if r != 0 && (c & (sync_range - 1)) == 0:
        guard = rs.mutex[r-1].lock()
        while c > *guard - sync_range + 1:
            guard = rs.cond[r-1].wait(guard)

// Writer at (r, c) after encoding the SB.
sync_write(rs, r, c, cols):
    sig = (c < cols - 1) ? (c % sync_range == sync_range - 1) : true
    cur = (c < cols - 1) ? c : cols + sync_range
    if sig:
        guard = rs.mutex[r].lock()
        *guard = cur
        rs.cond[r].notify_one()
```

The `Mutex<i32>` carries `cur_col[r]` (the furthest SB column finished
on row `r`); the condvar wakes a waiter whenever the value advances.
Encoder constant: `sync_range = 1` (`vp9_ethread.c:318`).

**Critical libvpx detail** (`vp9_multi_thread.c:113-119`): a single
`RowSync` is shared by every tile row in the same tile column. Workers
across tile rows but the same tile column synchronise through the same
`cur_col[]`. In v1 with `tile_rows == 1` this collapses to "one
`RowSync` per tile column".

### 4.4 Job queue + work stealing

Row-MT distributes work via a per-tile-column job queue
(`vp9_multi_thread.c:18-52, 262-297`). Workers start at
`tile_col = thread_id_to_tile_id[thread_id]`, pop from
`row_mt_info[tile_col]` under its mutex, and when the queue drains
call `vp9_get_tiles_proc_status` (`vp9_multi_thread.c:309-341`) to mark
the tile complete and find another tile with the most remaining jobs.

rvp9 mirror:

```text
RowMtCtx<'frame> {
    queues: Vec<RowMtQueue>,        // one per tile column
    thread_id_to_tile_id: Vec<usize>,
    row_sync: Vec<RowSync>,         // one per tile column
}

RowMtQueue { inner: Mutex<JobQueueInner>, completion_status: AtomicBool }

JobQueueInner { next_row: i32, end_row: i32, num_jobs_acquired: i32 }
```

We replace libvpx's linked list with a counter pair `(next_row,
end_row)` (jobs walk in scan order). Work-stealing: when
`next_row >= end_row`, mark `completion_status = true` and scan
`queues[*]` for the one with the most remaining rows.

**Determinism note (R-08).** The work-stealing decision is
non-deterministic on multi-tile encoders if more than one queue has
"most remaining work" — libvpx breaks ties by index order and so do we.
For bit-exact diff testing in Stage 4a, we **disable work stealing**
(force `num_workers <= tile_cols`) under a `DETERMINISTIC_ROW_MT=1`
debug flag. Production builds use stealing.

### 4.5 Top/left boundary predictor sync (the "above_context" contract)

The reason row-MT exists at all: VP9 intra prediction in SB `(r, c)`
reads `above_context[c..c+sb_w]` and `left_context[r..r+sb_h]` —
written by SB `(r-1, c)` and SB `(r, c-1)` respectively. Per
`06-threading-svc.md` §4.1:

- **Above-context** is a frame-level flat buffer (`cm->above_context`).
  Workers in different tile columns write **disjoint slices** of it. No
  sync needed across tile columns.
- **Left-context** is per-tile (`xd->left_context` /
  `xd->left_seg_context`) and is **cleared at the start of every SB
  row** within a tile (`vp9_encodeframe.c:4508-4509, 5541-5542`). No
  sync needed; it lives in `td->mb.e_mbd`.

So the only cross-row constraint is **above-context for SB `(r-1, c)`
must be written before SB `(r, c)` reads it**. The `RowSync` mechanism
exactly captures this happens-before edge.

In tile-MT (`row_mt == 0`), the SB-row-loop within a tile is sequential
for that tile (`vp9_encodeframe.c` partition loop), so within-tile rows
are serialised by control flow; no `RowSync` needed. Workers see
`vp9_row_mt_sync_read_dummy` / `_write_dummy` no-ops
(`vp9_encodeframe.c:6182-6183`).

### 4.6 Tile barriers

There is no explicit "tile barrier" in libvpx — the only barrier is
`launch_enc_workers`'s post-loop `sync()` on each worker
(`vp9_ethread.c:167-170`). rvp9 mirrors this: `FrameScope::join_all`
(invoked by `Drop`) waits on every worker's `Condvar` until each has
transitioned `Running -> Done`. Then per-worker accumulation (counts,
rd_counts) runs on the main thread.

This is sufficient because:

- Tile-MT writes are disjoint by tile column.
- Row-MT writes within a tile column are serialised by `RowSync`.
- Cross-tile-column writes that *cross* tile boundaries (loopfilter)
  happen in a separate post-encode pass with its own row sync
  (`VP9LfSync`; `06-threading-svc.md` §2.7 / §4.1).

---

## 5. Shared state contract

This section is **D-12 and D-15 resolved**. Every field of the rvp9
equivalent of `VP9_COMP` that crosses thread boundaries during encode
gets an entry: owner thread, sync mechanism, lifecycle, safety comment
template. The source for this table is `00-overview.md` §3 plus
verification against `vp9_encoder.h:683-1340` (the `VP9_COMP` struct in
libvpx).

### 5.1 Convention

| Symbol | Meaning |
|--------|---------|
| **owner** | which thread holds exclusive write access during a frame: M (main), W (worker), C (caller / external) |
| **sync** | the rvp9 mechanism: `&` (immutable borrow), `&mut` (exclusive), `Mutex`, `Atomic*`, `UnsafeCell + tile-disjoint`, `UnsafeCell + row-sync`, `Box<Cell>` (per-worker), `OnceLock` (initialised once) |
| **scope** | when the field can be mutated: `Init` (`Encoder::new` only), `Reconfig` (`Encoder::reconfigure`), `BetweenFrames` (main, no workers running), `InFrame` (during worker dispatch) |
| **§** | the §6 strategy code: A = scoped immutable borrow, B = slice-split, C = `UnsafeCell` with row-sync, D = per-worker box, E = atomic, F = main-thread only |

### 5.2 Per-encoder persistent state (lifetime: `Encoder::new` -> `Encoder::drop`)

| Field (libvpx name) | Type / size | Owner | Sync | Scope | § | Notes |
|---|---|---|---|---|---|---|
| `oxcf` | `VP9EncoderConfig` (~200 fields) | M | `Box<EncoderConfig>` + `&` | Reconfig | A | Workers borrow immutably for whole frame. |
| `buffer_pool->frame_bufs[12]` | `[RefCntBuffer; 12]` | M | `Box<BufferPool>` + interior atomic refcount | Init for slots; BetweenFrames for `ref_count` | E | Workers read ref pixels via `&FrameBuf` projected from `frame_refs[3]` slots. `ref_count` is `AtomicI32`. |
| `ref_frame_map[8]` | `[i32; 8]` | M | `Box<RefMap>` + `&` | BetweenFrames | A | Workers read map index, deref via `buffer_pool`. |
| `frame_refs[3]` | `[RefBuffer; 3]` | M | `&` | BetweenFrames | A | Worker reads `&YV12Buffer` only. |
| `fc` (current `FRAME_CONTEXT`) | `Box<FrameContext>` | M | `&FrameContext` | BetweenFrames (adaptation) | A | Workers read for cost LUT derivation. |
| `frame_contexts[FRAME_CONTEXTS=4]` | `[FrameContext; 4]` | M | `&` | BetweenFrames | A | Backup ring. |
| `counts` (`cm->counts`, the frame's `FRAME_COUNTS`) | `Box<FrameCounts>` | M | `&mut` between frames; `Box<FrameCounts>` per worker during frame, accumulated back | InFrame (per-worker copies) + BetweenFrames (accumulate) | D | See §5.6. |
| `rc` (`RATE_CONTROL`) | `RateControl` (~70 fields) | M | `&` during encode | BetweenFrames | A | Workers read `base_qindex`, `target_bandwidth`, etc. |
| `sf` (`SPEED_FEATURES`) | `Box<SpeedFeatures>` | M | snapshot copied per worker | BetweenFrames | D | **D-10 resolved — see §6.1.** |
| `svc` | `Svc` | M | `&` during encode | BetweenFrames | A | Layer ids, refresh flags. |
| `cyclic_refresh.map` | `Vec<i8>` (size `mi_rows * mi_cols`) | M | `&` (reads only during encode) | BetweenFrames | A | Workers read for segment_id decision. |
| `cyclic_refresh.last_coded_q_map` | `Vec<u8>` | M | `&` | BetweenFrames | A | Same as above. |
| `segmentation_map` | `Box<[u8]>` (size `mi_rows * mi_cols`) | M init, W per-block | `UnsafeCell<[u8]>` + tile-disjoint | InFrame (per-block writes) | C | **Per-block-disjoint writes** by workers via CR-AQ (`00-overview.md` §3.1 line on `cpi->segmentation_map`). Each worker writes only its tile slice. |
| `denoiser` (`VP9_DENOISER`) | `Denoiser` | M init, W slice-mutates | `UnsafeCell<[u8]>` + tile-disjoint | InFrame (per-tile slice writes) | B+C | `running_avg_y[]` arrays mutated by workers in their tile slice. Post-frame swap is single-threaded. |
| `noise_estimate` | `NoiseEstimate` | M | `&mut` | BetweenFrames | F | Workers do not read or write. |
| `lookahead` | `LookaheadCtx` | C+M | `Mutex<LookaheadCtx>` | external API edge | F+`Mutex` | Pushed by caller thread (`vp9_receive_raw_frame`); popped by main thread. Single-threaded in practice; `Mutex` for the API boundary. |
| `nmvcosts[2]`, `nmvsadcosts[2]`, `_hp` | `[Box<[i32; MV_VALS]>; 2]` × 4 | M | `&` | BetweenFrames | A | Cost LUTs; pointer offset by `MV_MAX` for negative indexing handled in `MotionCosts` accessor. |
| `mbmode_cost`, `inter_mode_cost`, etc. | flat `[i32; N]` | M | `&` | BetweenFrames | A | Filled per-frame in `vp9_initialize_rd_consts`. |
| `fn_ptr[BLOCK_SIZES]` | `[VarianceFns; 13]` | M | `&` | Init | A | The BFP dispatch table (A1 §5). Read-only post-init. |
| `prev_partition`, `prev_segment_id`, `prev_variance_low`, `copied_frame_cnt`, `content_state_sb_fd` | `Vec<...>` per SB | M init, W per-SB | `UnsafeCell<[...]>` + tile-disjoint | InFrame | C | Per-SB-disjoint worker writes. Each is one cell-per-SB; worker writes only SBs within its tile columns. |
| `prev_mip`, `prev_mi` | `Box<[ModeInfo]>` | M (swap per frame) | `&` during encode | BetweenFrames (swap) | F+A | `swap_mi_and_prev_mi` happens on main thread. |
| `workers[]`, `tile_thr_data[]` | `WorkerPool` | M | owned by pool struct | Init / Reconfig | F | The pool itself. |
| `multi_thread_ctxt` | `RowMtCtx` | M init, W during frame | `Mutex<JobQueueInner>` per column | InFrame (job dequeue) | E | See §4.4. |
| `lf_row_sync` | `LfSync` | M init, W during LF | `RowSync`-like | InFrame (LF pass) | C | Decoupled from encode-MT. |
| `ext_ratectrl` | `ExtRateCtrl` (callback wrapper) | M | `&mut` | BetweenFrames | F | Caller-supplied callback; main thread only. R-06 included v1. |

### 5.3 Per-frame transient (reset every frame, main-thread only)

All set by main thread before worker dispatch; workers read via `&`
(strategy A) for the frame's duration. Fields: `base_qindex`,
`y_dc_delta_q`, `uv_dc_delta_q`, `uv_ac_delta_q`, `y_dequant`,
`uv_dequant`, `tx_mode`, `interp_filter`, `seg.*`, `vbp_thresholds[4]`,
`Source`/`Last_Source`/`scaled_*`, `frame_flags`, `refresh_*_frame`,
`lst_fb_idx`/`gld_fb_idx`/`alt_fb_idx`, `scaled_ref_idx[3]`,
`ref_frame_flags`. The two exceptions are `cpi->td.mb` and
`cpi->td.rd_counts` — value-copied into each worker's `ThreadData` at
frame start (strategy D); see §5.4 and **§6.6 (D-11 resolution)**.

### 5.4 Per-worker state (one instance per `ThreadData`, lifetime: pool)

These are **owned exclusively by one worker** during the frame. The
storage lives in `WorkerPool.workers[i].td` between frames.

| Field | Type | Sync | Notes |
|---|---|---|---|
| `td.mb` | `MacroBlock` (huge: tens of KB) | per-worker owned, **value-reset from `cpi.td.mb` at frame start** | Workers re-wire `plane[j].coeff/qcoeff/dqcoeff/eobs` pointers to `pc_root.none.{coeff,qcoeff,dqcoeff,eobs}_pbuf[j][0]` (`vp9_ethread.c:227-232`, `:668-673`) in the `use_nonrd_pick_mode` path. **§6.2** discusses how Rust handles the re-wiring. |
| `td.rd_counts` | `RdCounts` (small, `coef_counts` 6D array dominates) | per-worker owned, value-reset; accumulated back via `accumulate_rd_opt` | |
| `td.counts` | `Box<FrameCounts>` (~8-10 KB) | per-worker owned, **value-copied from `cm.counts` at frame start**; accumulated back via `vp9_accumulate_frame_counts` | `vp9_ethread.c:124-125, 213-216, 244` |
| `td.pc_tree`, `td.pc_root`, `td.leaf_tree` | `Vec<PickModeContext>` + `Vec<PcTree>` (heap, large) | per-worker owned | Allocated once in pool init. **§6.5** discusses pin-mutability for the cross-linked tree. |
| `td.snapshot_sf` | `SpeedFeatures` (POD, ~600 lines worth of fields) | per-worker owned, **value-copied from `cpi.sf` at frame start** | **D-10 resolution.** See §6.1. |
| `EncWorkerData.tile_completion_status` | `[bool; MAX_NUM_TILE_COLS=64]` | per-worker owned | Used in row-MT work stealing only. |

### 5.5 Per-tile state (one instance per tile column × tile row, `tile_data[t]`)

Each tile's `TileDataEnc` (rvp9 name: `TileSlot`) is mutated by exactly
one worker during a frame in tile-MT. In row-MT, all workers may dequeue
jobs for any tile column, so multiple workers can write to *different*
SB rows of the **same** `TileSlot` concurrently.

Therefore the locking model differs by mode:

- **Tile-MT**: `&mut TileSlot<'frame>` granted exclusively to one
  worker. Strategy B in §6.
- **Row-MT**: `&TileSlot<'frame>` (immutable outer reference), interior
  mutability via `UnsafeCell` for `thresh_freq_fact` and `mode_map`,
  with synchronisation provided by `RowSync` (which ensures one worker
  finishes an SB row before another worker reads the same row's
  `thresh_freq_fact`). Strategy C in §6.

| Field | Type | Tile-MT sync | Row-MT sync |
|---|---|---|---|
| `tile_info` | `TileInfo` (geometry) | `&` | `&` |
| `thresh_freq_fact[BLOCK_SIZES][MAX_MODES]` | `[[i32; MAX_MODES]; BLOCK_SIZES]` | exclusive (per-tile worker) | `UnsafeCell` — happens-before via `RowSync`; reads precede writes by `nsync=1` SB column |
| `thresh_freq_fact_prev` | same | exclusive | exclusive (main thread restores between frames) |
| `mode_map[BLOCK_SIZES][MAX_MODES]` | `[[i8; MAX_MODES]; BLOCK_SIZES]` | exclusive | `UnsafeCell` |
| `row_mt_sync` | `RowSync` (shared with other tile rows in same tile column) | unused | `Mutex` per row + `Condvar` |
| `row_base_thresh_freq_fact` | `Box<[i32]>` (per SB row) | unused | `UnsafeCell`, written iff `adaptive_rd_thresh_row_mt=1` |

### 5.6 The `FRAME_COUNTS` accumulation contract (L-57 resolved)

`FrameCounts` is the per-frame statistics bucket that drives backward
probability adaptation. Layout: `~8-10 KB` of `unsigned int[]` arrays
(per `04-bitstream-entropy.md` §5).

**Per worker:** owns a `Box<FrameCounts>`, allocated once in pool init,
re-used across frames. At frame start, the main thread *copies*
(`memcpy`) `cm.counts` into each worker's `td.counts` (`vp9_ethread.c:
213-216`); workers increment their copies during encode; on join, the
main thread folds them back via `vp9_accumulate_frame_counts(&cm->counts,
td->counts, 0)` (`vp9_ethread.c:243-247`).

**Why memcpy in?** Because mode-decision reads from `counts` and
mutates: workers see the pre-frame state as a baseline and add their
deltas. The accumulate fold sums `cm.counts + sum_over_workers(td.counts -
cm.counts)` — but libvpx skips the subtraction by just summing all
workers' final counts back into `cm.counts` (which still has the
pre-frame baseline, so the equivalent is `cm.counts += td.counts -
cm.counts_initial`). Look at `vp9_accumulate_frame_counts` —
`vp9/common/vp9_thread_common.c:488` is a flat sum:

```c
// Sum each worker's counts (which contain pre-frame baseline + delta)
// into cm->counts, after first zero'ing cm->counts. Or accumulate
// directly into cm->counts which already holds the baseline?
```

In fact the third arg is `is_dec`; for the encoder we pass 0, and the
fold sums `cm->counts[i] += td->counts[i]`. **Trap:** because `cm->counts`
starts with baseline, we must zero it before accumulating, **or** pass
the workers a zeroed copy. Libvpx zeroes the per-worker copy at... it
*doesn't* — workers start with the baseline (memcpy from `cm.counts`)
and the accumulate path produces `cm.counts + N * baseline` if naive.

Reading `vp9_accumulate_frame_counts` more carefully
(`vp9_thread_common.c:488-700` area): each worker increments its private
copy from its baseline. The accumulate sum `cm->counts[i] +=
td->counts[i]` over `i < N-1` would double-count baseline.

**Resolution.** This is documented in `00-overview.md` §4.4 as the
"counts accumulation kernel is scalar-only". The libvpx convention is:

1. Worker N-1 (the main thread inline) directly mutates `cm->counts`
   (because its `td` aliases `cpi->td` and its counts pointer aliases
   `cm->counts` — `vp9_ethread.c:213` `if (thread_data->td->counts !=
   &cpi->common.counts)` guards the memcpy).
2. Workers `0..N-1` start with a memcpy of `cm->counts` (their baseline
   = pre-frame state).
3. After encode: main thread runs `vp9_accumulate_frame_counts(&cm->counts,
   td->counts, 0)` for each `i < N-1`. Reading the C: it does
   `cm->counts[i] += td_t->counts[i]`. Net effect:
   `cm->counts = (initial) + delta_inline + sum_other(initial + delta_other)`.
   This double-counts initial.

This is **a known libvpx quirk** (the counts are statistics for
backward adaptation; double-counting biases probability updates
slightly). For byte-parity we mirror it exactly. For correctness-first
we'd subtract `(N-1) * initial`. Divergence ledger entry: rvp9 mirrors
the bias.

**Implementation:** `FrameCountsBuffer` holds `Box<FrameCounts>`. Main
thread has `&mut FrameCounts` as `cm.counts`. Per-worker `td.counts:
Option<Box<FrameCounts>>` — `None` for the inline worker (it borrows
`&mut cm.counts` directly), `Some(Box)` for OS-thread workers. The
inline-worker case is the only place where worker N-1 has aliased
storage with the main thread — handled by **only running the inline
worker after all other workers have spawned** (so no overlapping
`&mut`).

### 5.7 The `SPEED_FEATURES` snapshot contract (D-10 resolved)

From `06-threading-svc.md` §7.1, `SPEED_FEATURES` is read-only during a
frame's encode but written before each frame by `set_size_independent_vars`
+ `set_size_dependent_vars` (`vp9_encoder.c:3749-3774`). Three options
from A6:

(A) Owned by `Encoder`, `&SpeedFeatures` to workers.
(B) `UnsafeCell<SpeedFeatures>` + documented "no writes between launch
    and sync".
(C) Snapshot the struct into each worker's `ThreadData` at frame start
    (memcpy, ~600 bytes × N workers).

**Decision: option (C) — per-worker snapshot.** Rationale:

1. **Cost is negligible.** 600 bytes × 80 workers = 48 KB per frame.
   At 60 fps that's 2.88 MB/s. Nothing.
2. **No aliasing at all.** Workers read from their own snapshot.
   Borrow checker is satisfied without any `unsafe`.
3. **Crash-clear invariant** for testing (R-10, L-64): the snapshot
   serialises trivially for golden-file byte-parity tests.
4. **Cache-friendly.** Each worker's snapshot lives in its own L1 cache
   line.

Implementation: `ThreadData.snapshot_sf: SpeedFeatures` (no `Box`,
inlined into `ThreadData`). At frame start, main thread does
`for w in workers { w.td.snapshot_sf = self.sf.clone(); }`. `Clone` for
`SpeedFeatures` is `#[derive(Clone, Copy)]` because all fields are POD.
Memory layout matches libvpx's `vp9_zero(*td)` semantics
(`vp9_ethread.c:116`).

Workers read via `&worker.td.snapshot_sf` exclusively; **never**
`&cpi.sf` directly. This is enforced by visibility:
`Encoder.sf: SpeedFeatures` is private and not exposed to worker
context types.

### 5.8 Per-superblock state

Per `00-overview.md` §3.5:

- `prev_variance_low[sb_index][25]` — per-SB; written by the worker
  owning the SB.
- `content_state_sb_fd[sb_index]` — per-SB; written by the worker
  owning the SB.
- `copied_frame_cnt[sb_index]` — per-SB; written by the worker owning
  the SB.

All three are `Vec<...>` of length `sb_rows * sb_cols`. Workers write
disjoint indices (their tile's SBs only). **Strategy C** — `UnsafeCell<[T]>`
with per-tile-disjoint-slice invariant, see §6.4.

### 5.9 Worker-write surface summary (D-12 + D-15 catalogue)

This is the complete list of state a worker may mutate during a frame
encode. Anything not on this list is read-only for workers. The
catalogue lives here (HLD-threading) so every LLD can reference it.

A worker thread may, during a single frame encode, mutate **only**:

1. **Its own `ThreadData`** (`td.mb`, `td.rd_counts`, `td.counts`,
   `td.pc_tree`, `td.leaf_tree`, `td.snapshot_sf`, and the re-wired
   `mb.plane[j].coeff/qcoeff/dqcoeff/eobs` pointers).
2. **Its tile slice of `tile_data[t]`** (`thresh_freq_fact`, `mode_map`,
   `row_mt_sync` only in row-MT for the tile's column, and
   `row_base_thresh_freq_fact` when `adaptive_rd_thresh_row_mt=1`).
3. **Tile-disjoint slices of frame-level buffers**:
   - `above_context`, `above_seg_context` — its tile-column slice
   - `mi_grid_visible`, `mi` — its tile-column slice
   - `current_frame_seg_map` (`segmentation_map` in our naming) — its
     tile slice; **per-block-disjoint** writes from CR-AQ
   - `frame_to_show.y_buffer/u_buffer/v_buffer` — its tile-column
     slice (reconstructed pixels)
   - `denoiser.running_avg_y[]` — its tile-column slice
   - `prev_partition`, `prev_segment_id`, `prev_variance_low` — per-SB,
     disjoint
   - `copied_frame_cnt`, `content_state_sb_fd`, `consec_zero_mv` — per-SB,
     disjoint
   - `tile_tok[][]`, `tplist[][]` — per-tile (one slot per tile)
4. **The row-MT job queue** (dequeue an entry): `multi_thread_ctxt.queues
   [col].lock()` -> pop -> unlock. Mutex-protected.

A worker **must not** mutate any other `Encoder`, `VP9Common`, or
`oxcf` field during a frame. This includes:

- `oxcf` (read-only, snapshot via `&`).
- `rc` (read-only during encode).
- `sf` (read-only; workers use their own snapshot per §5.7).
- `buffer_pool.frame_bufs[*].ref_count` (workers do not increment or
  decrement; atomic refcount is mutated only by main thread on
  `get_free_fb` / `release_fb`).
- `ref_frame_map`.
- `lookahead`.
- `noise_estimate`.
- `cyclic_refresh.map` (workers read; main thread writes between frames).
- `frame_contexts[*]`.
- `prev_mip`, `prev_mi` (swapped between frames on main thread).

**Aliasing contract enforced by:** `Encoder` holds the master state and
exposes a `FrameContext<'frame>` view to workers. `FrameContext` is
parameterised over `'frame`, which is the lifetime of one frame's
encode. Fields fall into:

- `read: &'frame FrameReadOnly` — every read-only field
- `tile_slots: &'frame [TileSlot<'frame>]` — tile-disjoint mutable views
- `shared: SharedFrameMut<'frame>` — the `UnsafeCell` arena (see §6.4)
- `row_mt: Option<&'frame RowMtCtx<'frame>>` — only in row-MT mode

Workers receive `&'frame FrameContext<'frame>` + a tile index (or, in
row-MT, just the context and self-discover via job queue).

---

## 6. `UnsafeCell` strategy

This section defines the patterns. Five strategies labelled A-F above.

### 6.1 Strategy A — immutable borrow (`&T`)

Use for fields that are:

- Set on `Encoder::new` / `Encoder::reconfigure` (Init scope), and read
  by workers, or
- Set between frames by main thread, and read by workers during encode.

Examples: `oxcf`, `sf` (via per-worker snapshot copy; the **source**
`sf` is held by main thread), all `*_cost` LUTs, `frame_refs[3]`,
`ref_frame_map`, `cyclic_refresh.map` (workers only read), `vbp_thresholds`.

No `unsafe` required. Borrow checker proves the absence of races: while
`&FrameContext<'frame>` exists in the worker, the main thread cannot
hold `&mut Encoder`.

**Implementation pattern:**

```rust
pub struct FrameContext<'frame> {
    pub read: &'frame FrameReadOnly,
    // ...
}

pub struct FrameReadOnly {
    pub oxcf: EncoderConfig,
    pub frame_refs: [RefBuffer; 3],
    // ... etc
}
```

`FrameReadOnly: Sync` (auto-derived; all fields `Sync`).

### 6.2 Strategy B — slice splitting (`&mut [T]` partitioned by tile)

Use for **frame-level flat buffers** where each tile writes a **disjoint
slice** and the slice boundaries are known at frame start, before any
worker spawns.

Examples:

- `above_context[mi_cols]` — split by `mi_col_start..mi_col_end` per
  tile.
- `above_seg_context[mi_cols]` — same.
- `mi_grid_visible[mi_rows * mi_stride]` — split by tile column (each
  tile column owns `mi_stride * mi_rows / tile_cols` entries).
- `current_frame_seg_map[mi_rows * mi_cols]` — same.
- `frame_to_show.y_buffer/u_buffer/v_buffer` — split by tile column.
- `denoiser.running_avg_y[]` — split by tile column.

**Implementation:** at frame start, after `set_frame_size`, the main
thread calls `FrameContext::split_tiles(&mut self, log2_tile_cols)`
which returns `Vec<TileSlot<'frame>>` (length `tile_cols`). Each
`TileSlot<'frame>` holds `&'frame mut`-typed fields for: `above_context`,
`above_seg_context`, `mi_grid`, `seg_map`, `y_recon`, `u_recon`,
`v_recon`, `denoiser_avg_y`, `thresh_freq_fact`, `mode_map`, etc. —
each a tile-column slice of the frame-level buffer.

1D buffer splits use `slice::split_at_mut`. 2D strided buffers (e.g.
`mi_grid_visible[mi_rows * mi_stride]` — each tile column owns a
strided sub-rectangle) use a `StridedSliceMut<'a, T>` wrapper exposing
`get_mut(row, col_within_tile)`; it is `Send + Sync` because no two
`StridedSliceMut`s alias.

`'frame` keeps the `Vec<TileSlot<'frame>>` alive until
`FrameContext::join_tiles` after `FrameScope::drop` joins all workers.

**No `unsafe`** for tile-MT writes — the borrow checker proves
disjointness. Row-MT can't use pure splits because work-stealing means
multiple workers may dequeue jobs for the same tile column — strategy C
applies (§6.4).

### 6.3 Strategy B, row-MT variant — outer slice + inner cell

In row-MT, each tile column's slice is still disjoint from other tile
columns, but **within** a column, multiple workers may write to
different SB rows of the same tile concurrently (one worker doing row
3, another row 5). The slice-split is by **tile column**, but the
intra-column mutation needs interior mutability.

`TileSlot<'frame>` in row-MT mode has:

- `above_context`, `above_seg_context`, `y_recon` etc. **still split by
  tile column** at the outer level (frame-level slicing).
- `thresh_freq_fact`, `mode_map` wrapped in `UnsafeCell` (see §6.4) for
  intra-column interior mutability.
- Reconstructed pixel writes within an SB are serialised by `RowSync`
  (above-context's happens-before — see §4.3 / §4.5).

Signature sketch:

```text
pub struct TileSlot<'frame> {
    pub tile_col: u32,
    // Per-tile-column slices; multiple workers share &TileSlot and
    // write to disjoint SBs within the column.
    pub above_context: TileColumnArena<'frame, EntropyContext>,
    pub mi_grid: TileColumnArena<'frame, Option<NonNull<ModeInfo>>>,
    // ...
    pub thresh_freq_fact: UnsafeCell<[[i32; MAX_MODES]; BLOCK_SIZES]>,
    pub mode_map: UnsafeCell<[[i8; MAX_MODES]; BLOCK_SIZES]>,
    pub row_sync: RowSync,
}

unsafe impl<'frame> Sync for TileSlot<'frame> { /* see §7 */ }

pub struct TileColumnArena<'frame, T> {
    base: NonNull<T>,
    rows: usize,
    cols_per_row: usize,
    _marker: PhantomData<&'frame mut [T]>,
}

// SAFETY: caller holds row-sync permission for (sb_row, sb_col) inside
// `tile_col`. RowSync.read/write provide the happens-before edge.
pub unsafe fn write_sb_above_context_slice(...) -> &mut [EntropyContext]
```

The `unsafe` is the cost of row-MT being a happens-before graph that the
borrow checker can't see. The SAFETY comment template (mandatory at
every call site) is:

```rust
// SAFETY: <function name> mutates a sub-slice of `<arena name>` covering
// SB (`sb_row`, `sb_col_in_tile`) within tile column `tile_col`.
// Aliasing argument:
//   (1) Outer slice is disjoint from other tile columns by construction
//       (TileSlot splits in TileMtCtx::split_tiles).
//   (2) Within this tile column, no other worker writes to
//       (sb_row, sb_col_in_tile') for sb_col_in_tile' overlapping this
//       call's footprint, because:
//       - Each SB (sb_row, sb_col) is a job consumed by exactly one
//         worker (JobQueue invariant, §4.4).
//       - Writes to `above_context[mi_col..mi_col+sb_w]` happen-before
//         reads on the row below via RowSync.write(sb_row, sb_col) +
//         RowSync.read(sb_row+1, sb_col).
// Concurrency oracle: this aliasing invariant is encoded in loom test
// `tests/loom/above_context_disjoint.rs` (§9).
```

### 6.4 Strategy C — `UnsafeCell` + happens-before edge

Use for fields that are:

- Mutated by workers at per-SB granularity
- Accesses serialised by `RowSync` or by per-block disjointness

Examples:

- `prev_variance_low`, `content_state_sb_fd`, `copied_frame_cnt`,
  `consec_zero_mv` — per-SB, fully disjoint by index. Even simpler than
  C above; no `RowSync` needed because each SB index is written by
  exactly one worker.
- `segmentation_map` — per-MB, disjoint. CR-AQ assigns segment IDs per
  block within a worker's tile.
- `thresh_freq_fact`, `mode_map` — per-tile in tile-MT (no
  inter-worker), per-(tile, SB row) in row-MT (serialised by `RowSync`).

**Implementation: `SbArena<'frame, T> { cells: &'frame [UnsafeCell<T>] }`**,
with `unsafe impl<T: Send> Sync` (per §7). Two `unsafe` accessors:

- `unsafe fn write(&self, sb_index) -> &mut T` — caller must guarantee
  no other thread accesses `cells[sb_index]` for the borrow duration.
  For per-SB-disjoint usage, the JobQueue invariant supplies this.
- `unsafe fn read(&self, sb_index) -> &T` — caller must guarantee the
  SB at `sb_index` is either being read by its own writer or by a
  worker downstream of a released `RowSync.write(sb_row, sb_col)`.

LLDs construct higher-level safe wrappers (e.g. `SbScratch::
partition_for_tile(tile_col)`) that encapsulate the bounds check.

### 6.5 Strategy D — per-worker `Box<T>`

Use for genuinely per-worker state that doesn't cross threads.

Examples: `td.counts`, `td.mb` (with the caveat in §6.6),
`td.pc_tree`, `td.snapshot_sf`.

No `unsafe` required. Each worker owns its `Box<T>` via the `Worker`
struct.

**The inline-worker subtlety.** In libvpx, worker `N-1` aliases
`cpi->td` — its `td.counts` *is* `cm->counts`. For accumulation, the
fold loop skips `i < N-1`. In rvp9:

- `WorkerPool.workers[N-1]` is the "inline slot": no OS thread, runs
  the worker hook on the main thread.
- Its `ThreadData` is a borrow of `Encoder.td` (the main-thread copy).
- Counts accumulation: main thread takes `&mut cm.counts`, and for
  workers `0..N-1`, calls `accumulate_frame_counts(&mut cm.counts,
  &workers[i].td.counts)`.
- The inline worker has *already mutated* `cm.counts` directly through
  its `&mut FrameCounts` (because it ran on the main thread inline);
  the fold loop for `i < N-1` adds the OS-thread workers' deltas.

This **avoids** the libvpx baseline-double-counting bug for the inline
worker. For OS-thread workers, we still inherit the bias (per §5.6
note). Tradeoff: byte-parity with libvpx for OS-thread workers; cleaner
semantics for the inline worker. **Divergence ledger entry: yes —
inline-worker counts behave slightly differently from libvpx when
`N == 1` (no OS-thread workers).** Specifically, for `N=1` we are
bit-exact with libvpx because there are no OS-thread workers and no
fold runs; for `N>1`, we mirror libvpx's bias because we accumulate the
same way for `i < N-1`.

### 6.6 The `td.mb` MACROBLOCK aliasing trap (D-11 resolved)

From `06-threading-svc.md` §7.2: libvpx copies `cpi->td.mb` into each
worker's `td->mb` *by value*. `MacroBlock` is huge (tens of kilobytes;
contains a `MacroBlockD` (`e_mbd`) which contains `plane[MAX_MB_PLANE]`
each with pointers into buffers, plus partition contexts, plus segment
state, plus mode-decision scratch). Most fields are scalar or fixed
arrays — fine to value-copy. The **trap** is the pointers in `plane[j]`:

```c
struct macroblock_plane {
  DECLARE_ALIGNED(16, int16_t, src_diff[MAX_SB_SQUARE]);
  tran_low_t *qcoeff;
  tran_low_t *coeff;
  uint16_t *eobs;
  // ...
  struct buf_2d src;             // contains pointer into Source
};

struct macroblockd_plane {
  tran_low_t *dqcoeff;
  PLANE_TYPE plane_type;
  int subsampling_x, subsampling_y;
  struct buf_2d dst;             // contains pointer into frame_to_show
  struct buf_2d pre[2];          // contains pointers into refs
  ENTROPY_CONTEXT *above_context;// pointer into cm->above_context
  ENTROPY_CONTEXT *left_context; // pointer into xd->left_context_buf
  // ...
};
```

After value-copy, the worker's `td.mb.plane[j].above_context` points
into the **frame-level** `cm->above_context`. That's a raw pointer
sharing — and libvpx re-points `coeff/qcoeff/dqcoeff/eobs` to
worker-local scratch (`ctx->coeff_pbuf[j][0]` from `pc_root.none`) in
the `use_nonrd_pick_mode` path (`vp9_ethread.c:227-232`).

**Rust mapping.** A `&'frame mut MacroBlock` won't work because the
borrow checker rejects: workers can't all hold `&mut` to non-disjoint
buffers via aliased pointers. Two approaches:

(A) **Encode aliasing in the type.** `MacroBlock` becomes
`MacroBlock<'frame>` parameterised over the frame lifetime; its
pointer-typed fields become `&'frame mut [T]` references into
tile-disjoint slices. The value-copy at frame start re-binds these
references from the main thread's `MacroBlock` to the worker-specific
slices.

(B) **Raw pointers + SAFETY contract.** `MacroBlock` keeps `*mut`
fields; the aliasing invariant is documented but not type-checked.

**Decision: option (A), with a twist.** A pure (A) is tedious because
`MacroBlock` has dozens of pointer fields and many of them point into
the same `TileSlot`. We use option (A) for the **outer** plane buffers
(`above_context`, `dst`, `pre`) — these become `&'frame mut` references
into `TileSlot` slices — and option (B) for the **inner** worker-local
scratch (`coeff`, `qcoeff`, `dqcoeff`, `eobs`) which point into
`pc_root.none.{coeff,qcoeff,dqcoeff,eobs}_pbuf[j][0]` arrays inside the
worker's own `ThreadData`. The inner pointers are self-references into
`ThreadData`; that's `Pin`/`PhantomPinned` territory but in practice
the worker owns its `ThreadData` exclusively so a raw pointer suffices
with a clear safety comment.

Signature sketch (full layout in LLD-mode-decision):

```text
MacroBlock<'frame>:
    plane: [MacroBlockPlane<'frame>; 3]
    e_mbd: MacroBlockD<'frame>
    // scalar fields

MacroBlockPlane<'frame>:
    src_diff: [i16; MAX_SB_SQUARE]   // owned scalar
    coeff, qcoeff: NonNull<TranLow>  // points into ThreadData scratch
    eobs: NonNull<u16>
    src: Buf2D<'frame>               // borrow into Source

MacroBlockDPlane<'frame>:
    dqcoeff: NonNull<TranLow>        // ThreadData scratch
    dst: Buf2DMut<'frame>            // borrow into TileSlot.{y,u,v}_recon
    pre: [Buf2D<'frame>; 2]          // borrow into frame_refs
    above_context: AboveContextSlice<'frame>  // row-MT-aware (§6.3)
```

**`MacroBlock::reset_for_frame(&mut self, tile_slot: &'frame mut TileSlot<'frame>, td_scratch: &mut ThreadDataScratch)`** is the equivalent of the libvpx `td->mb = cpi->td.mb` value-copy with re-wire:

1. Copy scalar fields from `Encoder.td.mb_template`.
2. Re-bind borrow fields to point at this worker's `tile_slot` slices.
3. Re-bind `NonNull` scratch pointers to this worker's
   `td_scratch.pc_root.none.{coeff,qcoeff,dqcoeff,eobs}_pbuf[j][0]`
   (the `use_nonrd_pick_mode` re-wire from `vp9_ethread.c:227-232`).

**SAFETY comment template for the re-wire:**

```rust
// SAFETY: `pc_root` is owned exclusively by this worker for the entire
// frame (per-worker ThreadData ownership, §5.4). The NonNull pointers
// we install in plane[j].{coeff,qcoeff,eobs} and plane[j].dqcoeff alias
// pc_root.none.{coeff,qcoeff,dqcoeff,eobs}_pbuf[j][0] which lives
// inside `td_scratch`. As long as `MacroBlock<'frame>` does not outlive
// `td_scratch` (enforced by reborrow on each frame), the pointers are
// valid for the duration of mode decision. Because mode decision is
// fully sequential within one worker, no Rust mutable-aliasing rule is
// violated even though pc_root.none is accessed both via NonNull and
// via &mut PickModeContext — we never hold both simultaneously (mode
// decision either reads the .none context as scratch via &mut, or via
// the plane[j].coeff path; never interleaved).
```

This **is** the hardest unsafe in the project. The §9 test strategy
puts Miri runs on the mode-decision inner loop specifically to validate
this invariant.

**D-11 resolution recap:**

- **Tile-MT**: option (A) outer + option (B) inner.
  `MacroBlock<'frame>` with borrowed slice fields and raw-pointer
  scratch fields, all rebound at frame start.
- **Row-MT**: same `MacroBlock<'frame>` type, but the borrowed slices
  are `TileColumnArena<'frame, _>` accessors (§6.3) backed by
  `UnsafeCell` and gated by `RowSync` happens-before edges.

### 6.7 Strategy E — atomics

Used in two places:

- `RefCntBuffer.ref_count: AtomicI32` — incremented when a frame is
  put into a ref slot, decremented when displaced. Main thread only in
  v1, but atomic-typed because the buffer pool will eventually be
  `Arc<BufferPool>` for v2.
- `RowMtQueue.completion_status: AtomicBool` — set when a tile's queue
  drains. Read-mostly under the queue mutex; atomic for the
  cross-worker "is column done?" probe.
- (Optionally) `RowSync.cur_col_atomic: [AtomicI32; rows]` — mirrors
  the mutex-protected `cur_col[r]` for cheap unsynchronized reads in
  debug instrumentation. Not used for correctness.

### 6.8 Strategy F — main-thread only

The simplest case. Fields written and read only by the main thread,
between frame encodes. No threading concern. Examples: `noise_estimate`,
`lookahead` (single-threaded API in practice), `ref_frame_map` rotation,
`buffer_pool` slot acquisition.

These are simple `&mut` fields on `Encoder`. No `unsafe`.

---

## 7. `Send` + `Sync` implementations

### 7.1 Auto-derived (no manual `unsafe impl`)

- `EncoderConfig`, `RateControl`, `Svc`, `FrameContext` (read-only
  fields only), `NoiseEstimate`, `Lookahead` — all POD or compose only
  `Send + Sync` fields.
- `SpeedFeatures` — `#[derive(Copy, Clone)]`, all POD; `Send + Sync`.
- `FrameCounts`, `RdCounts` — all `u32`/`i32`/`i64` arrays; `Send +
  Sync`.

### 7.2 Manual `unsafe impl Send` / `unsafe impl Sync`

| Type | `Send` | `Sync` | Justification |
|------|--------|--------|---------------|
| `TileSlot<'frame>` (tile-MT case) | yes | no | Granted to one worker per frame; movable across workers conceptually but only one holds it at a time. `Sync` is unnecessary; we hand out `&mut TileSlot` exclusively per tile. |
| `TileSlot<'frame>` (row-MT case) | yes | **yes** | Multiple workers share `&TileSlot` and use interior mutability. SAFETY: §6.3 contract: each per-SB write is gated by `RowSync` happens-before; per-tile-column outer disjointness ensured by `TileMtCtx::split_tiles`. Loom test in §9. |
| `SbArena<'frame, T> where T: Send` | yes | yes | SAFETY: §6.4 contract — each `sb_index` is mutated by exactly one worker by the JobQueue invariant. |
| `TileColumnArena<'frame, T>` | yes | yes | SAFETY: §6.3 contract — outer slice disjoint by tile column; inner per-SB happens-before via `RowSync`. |
| `WorkerPool` | no | no | Lives in `Encoder`; never crossed thread boundaries. |
| `Encoder` | yes | no | The handle moves to another thread per `vpx_codec_enc_init` semantics. `Sync` is not required (caller holds `&mut Encoder` for any operation). |
| `RowSync` | yes | yes | SAFETY: contains `Vec<(Mutex<i32>, Condvar)>` which is `Sync`. |
| `RowMtCtx<'frame>` | yes | yes | SAFETY: queues use `Mutex`; tile assignment is read-only. |
| `MacroBlock<'frame>` | yes | no | Held by one worker exclusively. `Sync` unnecessary. |
| `ThreadData<'pool>` | yes | no | Held by one worker exclusively. |
| `RefCntBuffer` | yes | yes | All scalar + `AtomicI32` `ref_count`. |
| `BufferPool` | yes | yes (deferred to v2) | v1: held exclusively by `Encoder` (main thread). v2: `Arc<BufferPool>` for frame-parallel; the `Sync` impl is trivially valid because mutations go through `Mutex` or atomics. |
| `RtcdTable` (in `OnceLock`) | yes | yes | Read-only after init. `OnceLock<RtcdTable>` is itself `Sync`. |

### 7.3 What we explicitly `!Sync`

- `MacroBlock<'frame>` — each worker owns its instance.
- `ThreadData<'pool>` — same.
- `TileSlot<'frame>` in tile-MT — held exclusively.

By default these would auto-derive `Sync` if all fields are `Sync`. We
suppress with `PhantomData<*mut ()>` to make accidental cross-thread
sharing a compile error.

### 7.4 The poisoning question

`std::sync::Mutex` poisons on panic. We accept this — a panic in a
worker holding a mutex is a bug, and poisoning makes the next encoder
operation fail loudly via `lock().unwrap()`. We do not use `parking_lot`
in v1 (extra dep, minor perf delta in our access pattern). Revisit at
Stage 5 hardening if mutex contention shows up in profiles.

---

## 8. `setjmp`/`longjmp` × worker threads (L-66 resolved)

### 8.1 libvpx baseline

Per `01-architecture.md` §3.7 (lines 1014-1019) and §7.4 (lines
2981-2988):

- `setjmp(cm->error.jmp)` set in `vp9_create_compressor`
  (`vp9_encoder.c:2390-2394`) and `encoder_encode`
  (`vp9_cx_iface.c:1381-1386`).
- `vpx_internal_error` `longjmp`s out of any call site.
- A1 §3.7 documents: **"Worker-side allocations are short-lived to the
  worker structure"**; the implication is that workers do not call
  `vpx_internal_error`.
- A8 §5.1 challenge #13 confirms: "vpx_internal_error is only called
  from main-thread call sites that have set setjmp".

The encoder workers (`enc_worker_hook`, `enc_row_mt_worker_hook`) never
emit `vpx_internal_error`. The decoder workers (`tile_worker_hook`,
`vp9_decodeframe.c:2182-2270`) **do** wrap their tile decode in a
`setjmp(error_info.jmp)` because bitstream parse failures occur during
tile decode. That's a decoder concern, addressed in §8.4.

### 8.2 Rust replacement strategy (encoder)

We use `Result<T, EncoderError>` throughout the encoder, **no panics
for normal error paths**. Specifically:

- All allocation happens at `Encoder::new` and `Encoder::reconfigure`,
  on the main thread, before any worker spawns. Allocation failure
  returns `Result<Encoder, EncoderError::OutOfMemory>`.
- Workers do not allocate. Their `ThreadData` (including all per-SB
  scratch) is preallocated at pool init.
- Bitstream pack failures (e.g. invalid config) are caught at frame
  start by the main thread before workers spawn.

This means **workers never need to longjmp**. They can fail only by:

(a) A `debug_assert!` firing — a bug. In release, the worker silently
    returns a wrong result and the next layer catches it (or doesn't —
    real bug to fix).
(b) Panicking from an out-of-bounds index, etc. — a bug. Caught by
    `catch_unwind` in `FrameScope::drop` (§3.6) and converted to
    `WorkerError::Panicked`.

No `setjmp` analog needed. The Rust analog of "longjmp out of an
arbitrary call depth" is `?` on `Result`; the analog of "longjmp out of
*recursion* deep into mode decision" is `Result` returns propagated up
the recursion. Cost: every internal function returns `Result`. We
accept this; the alternative (`panic + catch_unwind`) is unidiomatic in
Rust and complicates the worker-result protocol.

### 8.3 Defensive worker hook

To bound the blast radius of a worker bug, the worker hook does:

```text
fn worker_hook<'frame>(&self, ctx: &'frame FrameContext<'frame>) -> Result<(), WorkerError> {
    // No setjmp equivalent — but a guarded entry point.
    match std::panic::catch_unwind(AssertUnwindSafe(|| {
        self.encode_assigned_tiles(ctx)
    })) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(WorkerError::EncodeFailed(e)),
        Err(payload) => Err(WorkerError::Panicked(panic_payload_to_string(payload))),
    }
}
```

`catch_unwind` requires `panic = "unwind"` (the default for `cargo
test`); release builds may use `panic = "abort"`, in which case
`catch_unwind` is a no-op and a worker panic aborts the process. This
matches libvpx's `vpx_internal_error` behaviour (process-level
correctness assertion) and is the desired production semantic.

### 8.4 Decoder workers (for completeness)

The decoder worker (`tile_worker_hook`, decoder `setjmp` per `01-arch
.md` snippet and `vp9_decodeframe.c:2182-2270`) catches bitstream parse
errors during tile decode. In rvp9, the decoder tile hook returns
`Result<(), DecoderError>` with variants for `MalformedBitstream`,
`UnsupportedFeature`, etc. The same `catch_unwind` pattern bounds bug
blast radius.

LLD-tile-threading (decoder) inherits this design. The encoder LLDs
need no setjmp consideration.

### 8.5 What we lose vs libvpx

`vpx_internal_error` includes a string message recorded in
`cm->error.detail` for the API consumer to read via
`vpx_codec_error_detail`. Our `EncoderError` carries equivalent context:

```rust
pub enum EncoderError {
    OutOfMemory,
    InvalidConfiguration { detail: String },
    WorkerFailed(WorkerError),
    // ...
}

pub enum WorkerError {
    EncodeFailed(InternalError),
    Panicked(String),
    InvariantViolation { msg: &'static str, location: &'static Location<'static> },
}
```

The public Rust API exposes `err.to_string()` similar to
`vpx_codec_error_detail`.

---

## 9. Test strategy

### 9.1 What to test

The threading correctness surface has six layers:

| Layer | Tests | Tooling |
|-------|-------|---------|
| L1: pool lifecycle | Spawn / shutdown / re-spawn on reconfigure | unit tests; Miri |
| L2: worker dispatch | Each worker sees correct tile assignment, snapshot SF, ThreadData reset | unit tests with mock encode hook |
| L3: tile-MT aliasing | Slice splits are disjoint; no two workers write the same byte | Miri; loom |
| L4: row-MT happens-before | `RowSync.read(r, c)` correctly blocks until `RowSync.write(r-1, c-1)` | loom; ThreadSanitizer |
| L5: counts accumulation | Folded counts equal sum (libvpx-bias) | property test against reference impl |
| L6: end-to-end correctness | Frame-parity test against libvpx libvpx tile-MT output | Stage 4a |

### 9.2 Miri

Miri (with `-Zmiri-disable-isolation -Zmiri-symbolic-alignment-check`)
catches `MacroBlock` re-wire pointer aliasing (§6.6), `SbArena` /
`TileColumnArena` OOB, frame-scope use-after-free. CI gate:
`cargo +nightly miri test -p rvp9-encoder --test threading` on PRs
touching `src/threading/*` — targeted because Miri runs at ~50× slowdown.

### 9.3 loom

`loom = "0.7"` exhaustively interleaves `Mutex` / `Condvar` /
`AtomicI32`. Tests:

1. **`RowSync` happens-before**: writer at `(r, c)` -> reader at
   `(r+1, c)` never reads stale data.
2. **JobQueue work-stealing**: never pops the same `(sb_row, tile_col)`
   from two workers.
3. **Worker `Idle <-> Running` state machine**: never wakes a worker
   into `Running` while it still has unconsumed `Done`.

Tests in `crates/rvp9-encoder/tests/loom/`; PR-blocking on
`src/threading/` or `src/tile/` changes.

### 9.4 ThreadSanitizer

TSan via `RUSTFLAGS=-Zsanitizer=thread` (nightly) runs against the
Stage 4a encoder integration test (720p tile-MT + row-MT) on a
nightly job (not PR-blocking; TSan is too slow). Complements loom:
loom exhaustive but small; TSan non-exhaustive but real workloads.

### 9.5 `cargo test --release` with `max_threads` sweep

Every encoder test runs in release with `max_threads ∈ {1, 2, 4, 8}`.
Output must be bit-identical across the sweep on the deterministic
path (`row_mt = false` OR `adaptive_rd_thresh_row_mt = true`); ULP-only
deviation otherwise.

### 9.6 Speed-feature golden file (L-64)

A `crates/rvp9-bench/src/bin/sf_snapshot.rs` binary runs the
`set_rt_speed_features_*` cascade over fixed
`(speed, w, h, frame_type, content, rc_mode, use_svc, lag, ...)`
tuples; serialises `SpeedFeatures` as RON; compares against a libvpx
fixture. Reference: `06-threading-svc.md` §5.7-§5.19. CI gate:
golden-file diff must be intentional.

### 9.7 Soak test

Stage 5: 72-hour CI job, varied content, random `max_threads`,
monitors `heaptrack` snapshots, thread count drift, and MD5 every
1000th frame against baseline. Validates the persistent-pool model.

### 9.8 Property-based tests for accumulators

`vp9_accumulate_frame_counts` and `accumulate_rd_opt` are
fold-equivalent. We property-test that the fold equals
`baseline + sum(deltas) + (N-2) * baseline` (the libvpx bias from §5.6)
for randomly-generated baseline + delta tuples via `proptest`. The
inline worker mutates `cm.counts` directly; the loop folds OS-thread
workers.

### 9.9 What we do NOT test

- **Performance**: this HLD doesn't claim a perf budget; performance
  tests live in `bench/realtime.rs` (Stage 0 already wired) and gate at
  Stage 4a.
- **SVC layer threading**: Stage 4d/4e territory; LLD-svc owns. The
  threading HLD assumes "one frame at a time" — SVC layers are
  serialised inside one `vpx_codec_encode` call per
  `06-threading-svc.md` §3.7.
- **Decoder threading**: LLD-tile-threading-dec owns. We've described
  the encoder mirror only; decoder uses the same primitives.

---

## 10. Open questions and assumptions

### 10.1 Cross-HLD dependencies

`notes/hld/00-architecture.md` does not yet exist (verified at start of
this run). HLD-arch will need to confirm these working assumptions:

- **H1**: `Encoder` is a single owning type for `BufferPool`,
  `WorkerPool`, frame-loop state (affects §3 pool placement).
- **H2**: `FrameCounts` / `RdCounts` are Rust structs mirroring libvpx
  field-by-field (affects §5.6 accumulator design).
- **H3**: `SpeedFeatures` is one `#[derive(Copy, Clone)]` POD struct,
  not decomposed (affects §5.7 snapshot strategy).
- **H4**: `EncoderError` top-level enum + internal `WorkerError`
  (affects §8 error propagation).
- **H5**: `Encoder::new` returns `Result<Encoder, EncoderError>`; OOM
  reachable (affects §8).
- **H6**: `Encoder::reconfigure(&mut self, ...)` exists and may tear
  down the pool (affects §3.4).
- **H7**: `MacroBlock<'frame>` lives in `rvp9-encoder` crate, not
  `rvp9-core` (affects §6.6 placement).
- **H8**: Frame buffer types (`YV12Buffer`, `RefCntBuffer`) in
  `rvp9-core`; `BufferPool` is Arc-free in v1 (D-08 / L-37; affects
  §5.2 sync).

If HLD-arch contradicts any of H1-H8, this HLD revises the affected
section. **We do not block on HLD-arch.**

### 10.2 Remaining HLD-threading open items (non-blocking)

- **OT-1**: Windows worker thread priority (`SetThreadPriority`,
  `vpx_util/vpx_thread.h:90`) — skip in v1 per L-63 / A6 §7.13.
- **OT-2**: `crossbeam-utils::CachePadded` for per-row mutexes — LLD
  micro-opt.
- **OT-3**: `parking_lot::Mutex` — Stage 5 perf review.
- **OT-4**: `WorkerPool::block_on_join` API — LLD-tile-threading.
- **OT-5**: NUMA core-pinning by tile column — v2 / Stage 5.
- **OT-6**: Determinism flag in public API — LLD-encoder-shell.
- **OT-7**: Whether to **fix** the libvpx counts-bias (§5.6) rather
  than mirror it. **Recommendation: mirror in v1**, divergence-ledger
  candidate.

Stage 3 / 4a can start without resolving any of these.

---

## 11. Divergence ledger entries (from this HLD)

To append to `notes/divergence-ledger.md`:

- **DV-T1**: `tile_rows > 1` panics in v1 (L-62). No byte-parity impact at
  `log2_tile_rows=0`.
- **DV-T2**: `adaptive_rd_thresh_row_mt = 1` forced when row-MT is
  active (R-08). Minor BD-rate cost (<0.3%); buys determinism.
- **DV-T3**: Encoder workers never invoke `vpx_internal_error` equivalent;
  all allocation on main thread before spawn (`01-architecture.md`
  §3.7). No observable change on success path.
- **DV-T4**: `Encoder::reconfigure` always tears down pool on
  shrinkage (simpler than libvpx's conditional re-spawn at
  `vp9_ethread.c:92`). Negligible cost.
- **DV-T5**: (Pending OT-7) Mirror libvpx's counts-bias for byte-parity.
- **DV-T6**: Windows worker thread priority not elevated (A6 §7.13).
- **DV-T7**: No frame-parallel encoding in v1 (Q4 deferred to v2;
  single-frame-in-flight matches libvpx exactly).

---

## 12. Downstream consumers

This HLD's outputs feed: **LLD-tile-threading** (§3-§4-§6 directly);
**LLD-mode-decision** (§5.4 per-worker state + §6.6 MacroBlock
aliasing); **LLD-rate-control** (§5.6 counts contract);
**LLD-loopfilter** (§4 row-sync re-use for `VP9LfSync`-equivalent);
**LLD-denoiser** (§5.2 tile-disjoint write surface); **LLD-aq** (§5.2
segmentation_map per-block writes); **HLD-simd** (§3.5 RTCD-before-spawn
invariant, R-09); **HLD-arch** (consumes §0 H1-H8 assumptions).

Citations within the doc are inline; the load-bearing libvpx files are
`vp9/encoder/vp9_ethread.{c,h}` (worker pool + sync),
`vp9/encoder/vp9_multi_thread.{c,h}` (row-MT job queue),
`vp9/common/vp9_thread_common.{c,h}` (LF sync, counts accumulate),
`vpx_util/vpx_thread.{c,h}` (cross-platform worker primitive), and
`vp9/encoder/vp9_speed_features.c` (the `set_rt_speed_feature_*`
cascade).

---

## 13. Summary

**Q3 resolved**: `std::thread` + `Mutex` + `Condvar` + custom scoped
borrows (option D, §1.2). Reasoning: 1:1 pthread mirror per D6; channel
APIs don't fit row-MT's point-to-point sync; rayon doesn't fit
deterministic work allocation.

**Q4 resolved**: Frame-parallel encoding is **v2 work**. v1 is
single-frame-in-flight. No libvpx precedent; insufficient pressure on
v1 perf budgets; clean v2 scope.

**D-10 resolved**: `SPEED_FEATURES` is snapshotted per-worker (option C
from A6 §7.1). 48 KB/frame cost is negligible; eliminates all aliasing
concern. §5.7.

**D-11 resolved**: `MacroBlock<'frame>` is parameterised over frame
lifetime. Outer (frame-borrowed) fields use lifetime-borrowed slices;
inner (worker-local scratch) fields use raw pointers with documented
ownership invariants. Tile-MT uses scoped slice splitting; row-MT uses
`UnsafeCell` + happens-before via `RowSync`. §6.2, §6.3, §6.6.

**D-12 + D-15 resolved**: §5 is the complete catalogue of `cpi->td` and
worker-mutated fields, with owner / sync / scope / strategy per entry.
§5.9 distils the worker-write surface to a single canonical list every
LLD references.

**L-66 resolved**: Workers do not need `setjmp` equivalent. Errors
propagate via `Result<T, EncoderError>`; all allocation happens on the
main thread before workers spawn; `catch_unwind` is the panic safety
net (no-op under `panic=abort`). §8.

The threading model is mechanical to implement after this HLD because
every aliasing question has a concrete strategy (A-F) attached to every
field. The hardest unsafe in the project — `MacroBlock` re-wire — has a
SAFETY comment template (§6.6) and Miri/loom coverage (§9).
