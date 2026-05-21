# LLD: core-tile-threading

**Stage 3 / Wave C G5 LLD.** Implementation-level design for the
generic `rvp9-core::tile_threading` primitive — the [`WorkerPool`] +
[`FrameScope`] shape that the decoder and encoder both wrap. The
primitive is intentionally small (~250 lines including comments);
this LLD documents the contract so the decoder/encoder LLDs can
reference it instead of re-deriving the soundness chain.

**Date:** 2026-05-21.
**HLD anchor:** `wrk_docs/2026.05.19 - HLD - decoder hardening.md`
§G5 (lines 970-1310). V6 K-3 redesign (closure-form, not Drop-based).
**Reference implementation:** `crates/rvp9-core/src/tile_threading/`.

---

## 1. Purpose

A long-lived worker pool that survives across frames, plus a per-
frame "scope" handle for submitting work that may borrow stack data
from the caller. Generic — no decoder or encoder concepts leak in.

Concretely:

- [`WorkerPool::new(n: NonZeroUsize)`] — spawns `n` OS threads, each
  blocking on a `crossbeam_channel::Receiver<WorkerJob>`.
- [`WorkerPool::scope<F, R>(&mut self, f: F) -> R`] — opens a per-
  frame scope, runs `f` with `&FrameScope<'_, 'frame>`, blocks on
  every submitted job before returning.
- [`FrameScope::submit<F>(&self, job: F)`] where `F: FnOnce() + Send
  + 'frame` — enqueues a job. The `'frame` lifetime makes captures
  that borrow from the `scope()` caller's stack safe.
- `Drop for WorkerPool` — drops the job sender (workers see channel
  close, exit `recv` loops), then joins each handle.

---

## 2. Why closure-form, not Drop-based

V3 of the HLD proposed `let scope = pool.scope(); scope.submit(...);
drop(scope);` with a Drop impl that joined pending jobs. V6 K-3
spotted the soundness gap: `mem::forget(scope)` would skip the Drop,
leaving workers with `'frame`-bounded captures pointing at freed
stack data.

The closure-form `pool.scope(|scope| { ... })` API makes
`mem::forget` structurally impossible: the caller never sees an
owning `FrameScope` value, only a borrowed `&FrameScope`. The HRTB
`for<'frame> FnOnce(&FrameScope<'_, 'frame>) -> R` makes `'frame`
fresh per call, so the caller cannot smuggle the reference past the
closure boundary either.

This is the same pattern as `std::thread::scope`.

---

## 3. SAFETY contract

One `unsafe` site: [`FrameScope::submit`] transmutes the job closure
from `Box<dyn FnOnce + Send + 'frame>` to `Box<dyn FnOnce + Send +
'static>` so it can travel through the `crossbeam_channel::Sender<
Box<dyn FnOnce + Send + 'static>>` job queue.

Soundness chain (full version in
`crates/rvp9-core/src/tile_threading/SAFETY.md`):

- The `pending` atomic counter is incremented before each submit,
  decremented after each completion.
- `WorkerPool::scope` runs `f` under `catch_unwind`, then
  unconditionally drains `done_rx` until `pending == 0`.
- Therefore by the time `scope()` returns, every submitted job's
  user-closure body has finished — the `'frame`-bounded borrows
  are no longer accessed.
- `mem::forget(scope)` is impossible (closure-form API; no owning
  value).

---

## 4. Panic protocol (two failure surfaces)

Two distinct unwind sites: (i) the user `f` itself, (ii) any
submitted job. The protocol:

1. Both sites run under `catch_unwind`.
2. The drain loop runs unconditionally — submitted jobs are joined
   regardless of how `f` exits.
3. After drain:
   - If `f` panicked, re-raise its payload. Worker panics are
     dropped on the floor (closure failure is primary).
   - Else if any worker panicked, re-raise the FIRST-submitted
     worker panic.
   - Else return `f`'s value.

The per-scope `Mutex<Vec<(usize, Box<dyn Any + Send>)>>` collects
worker panic payloads tagged by submission index. The post-drain
priority sort picks the lowest index.

---

## 5. crossbeam-channel justification

`std::sync::mpsc::Receiver` is `!Sync` — one worker can hold the
receiver, but multiple cannot share it. The pool needs MPMC because
one job queue feeds many workers. `crossbeam_channel` provides
`Receiver: Sync + Clone`; each worker clones the receiver and
competes on `recv()`.

---

## 6. Test surface (Stage 3 G5)

`crates/rvp9-core/src/tile_threading/scope.rs::tests` — 11 unit
tests:

- Single job runs.
- Many jobs all join.
- Jobs can borrow frame data (via `'frame`).
- Single-job panic propagates.
- First-panic-by-submission-order wins.
- Closure panic dominates worker panic.
- Closure panic still joins workers.
- Single-worker pool works (N=1 degenerate).
- Pool reusable across scopes.
- Empty scope.
- Scope returns closure value.

`crates/rvp9-harness/tests/tile_threading.rs` — 4 integration
oracles (HLD V6 §G5 oracles #1, #2, #4):

- Conformance at N=2 (201/0/0 — matches N=1 today).
- Inter-thread-count determinism: `decode(N=1) == decode(N=2)` for
  every shown frame in the corpus.
- Multi-tile focused subset: same determinism check on the three
  canonical `tile_1xN` vectors.
- Construct + drop smoke at N ∈ {1, 2, 8}.

Nightly CI deferred (per HLD §G5): TSan run, 100× repeat-run,
libvpx-parity oracle (oracle #3).

---

## 7. Decoder wrap

The decoder constructs `WorkerPool` in `with_threads(n)` and stores
it as `Decoder::worker_pool`. Both `decode_intra_frame_inner` and
`decode_inter_frame` wrap their per-tile loop in
`self.worker_pool.scope(|scope| { ... })`.

### 7.1 G5.0 first cut (all-inline) — superseded by G5.1

`scope.submit(...)` was unused; every tile ran sequentially on the
caller thread. Conformance held tautologically at N=2 because the
N=2 path was structurally identical to N=1.

### 7.2 G5.1 dispatch (current) — per-tile fresh `Yv12Buffer`

**Eligibility gate** (both intra + inter loops):

- `tile_rows == 1` — VP9 column-only split. Tile pixel rectangles
  are vertically full-height but horizontally disjoint at MI
  granularity (8 luma px = 4 chroma px at 4:2:0). Tile_rows > 1
  falls back to the inline path (rare in conformance; documented in
  `notes/g5.2-tile-rows-followup.md`).
- `buffers.len() > 1` — degenerate 1-tile frames stay inline.
- `self.num_threads.get() > 1` — single-threaded knob always
  inline.

**Per-tile state** (each tile owns a fresh, scope-lifetime
allocation):

- `scratch: Yv12Buffer` — a fresh full-frame-sized YV12 buffer
  built via `Yv12Buffer::clone_geometry()`. Each tile writes its
  column range into its own scratch buffer; the rest stays zero.
- `seg_ids: Vec<u8>` — fresh frame-sized seg-id slab; tile writes
  only its `(mi_col_start..mi_col_end)` cells.
- `mv_grid: Vec<PrevFrameMvRef>` (inter only) — fresh frame-sized
  MV grid.
- `coef_counts / mv_counts / mode_counts` — fresh per-tile
  count banks (Box-allocated to avoid stack churn for the
  `FrameCounts` ~62 KB struct).
- `result: Option<Result<TileDecodeCtx, _>>` — filled by the
  worker, drained by the main thread post-scope.

The slot struct uses `Mutex<Option<T>>` for each field — the
worker `.lock().take()`s on entry, `.lock() = Some(...)`s on exit,
and the main thread `.into_inner()`s after scope joins. This keeps
the ownership handoff clean and panic-safe (a worker panic leaves
the slot's mutex in a recoverable state).

**Post-scope merge** (in submission order):

1. `self.coef_counts.add(&counts)` etc — additive merge of every
   field via `FrameCounts::add` / `CoefCountsBank::add` /
   `NmvCounts::add`.
2. `yv12.merge_tile_columns_from(&scratch, luma_col_start, luma_col_end)`
   — memcpy each tile's column range from the scratch buffer to
   the destination YV12, row-by-row.
3. `cur_segment_ids[..][col_range].copy_from_slice(...)` /
   `cur_mv_grid[..][col_range].copy_from_slice(...)` — same column-
   range merge for the frame-sized auxiliary arrays.

**Determinism**: merging in submission order — not completion
order — guarantees frame-to-frame and run-to-run bit-exactness.
Workers complete in arbitrary order, but the merge loop iterates
`slots.into_iter().enumerate()` so the additive folds happen in
the same sequence on every run.

**Why per-tile fresh `Yv12Buffer`?** The simpler design — give
each tile a column-windowed view into the parent buffer — runs
into Rust's `&mut` aliasing rules for column-wise splits. A
contiguous-bytes `&mut [u8]` slice that includes rows from
neighboring tiles unavoidably aliases across tiles. The standard-
library `split_at_mut` only splits linearly (rows), not columns.
The remaining options are (a) careful unsafe with raw pointers
and a soundness chain that's awkward to audit, (b) per-tile fresh
allocations + a post-scope memcpy merge. We chose (b) because the
allocation cost is modest (one ~3MB Yv12Buffer per tile per
frame for 1080p Yuv420; ~12MB total for 4 tiles), the soundness
proof is trivial (every byte is uniquely owned), and the merge
memcpy is a streaming sequential write that the OS allocator and
hardware prefetcher both handle well.

**Cost summary**: ~3 MB scratch / tile / frame allocation cost,
~3 MB merge memcpy / tile / frame. For tile-rows-1 vectors at
≤8 tiles this is a few percent of a single-tile decode's
wall-clock. Future G5.2 work can reduce this via a recycled
per-thread scratch buffer pool (`Decoder::scratch_pool`), but the
current shape is the floor that ships parallelism without
soundness debt.

**Instrumentation**: a process-wide `RVP9_G5_1_PARALLEL_JOBS`
atomic counter bumps inside every `scope.submit(...)` job body —
the harness regression test
`multi_tile_dispatch_uses_worker_pool` verifies the delta is
non-zero on `tile_1x2` to confirm the parallel path actually
fires (catches eligibility-gate misfires).

---

## 8. Why not just use `std::thread::scope`?

`std::thread::scope` spawns one OS thread per submission. The VP9
decoder needs a worker pool that survives across frames so the
per-frame thread-spawn cost is amortised. `std::thread::scope`'s
shape is otherwise nearly identical to ours; we reuse its
"closure-form, never-forgettable scope" technique.

---

## 8.1 Performance/cost notes

**Eager worker spawn (G5 first cut).** `WorkerPool::new(n)` spawns
`n` OS threads immediately and keeps them alive until `Drop`. This
is correct and simple, but has two reviewer-flagged cost concerns:

- **Test/fuzz thread amplification.** `Decoder::new()` defaults to
  `available_parallelism()` (commonly 8–32 on dev/CI hosts). Each
  `Decoder::new()` call spawns that many threads, so a unit-test
  file that constructs ~30 `Decoder`s spawns several hundred
  threads sequentially. The fuzz target builds a fresh `Decoder`
  per iteration, multiplying the spawn cost by iteration count.
  **Mitigation today:** test/fuzz sites that don't need parallelism
  use `Decoder::with_threads(NonZeroUsize::new(1).unwrap())`. The
  explicit single-thread path keeps test wall-clock predictable
  and avoids thread-scheduler noise inside dense unit suites.
- **Idle thread cost.** Even when a `Decoder` never calls
  `decode_frame`, the pool threads sit in `recv()` waiting for
  jobs. Each thread reserves a stack (~8 MB default on Linux),
  which on a 32-thread host is ~256 MB of address space per
  `Decoder` instance — fine for codec library callers, surprising
  for short-lived test fixtures.

**G5.1 deferred work.** Lazy worker spawning — workers created on
first `scope()` invocation rather than `new()`. The TODO comments
at `Decoder::new()`, `Decoder::with_threads()`, and
`WorkerPool::new()` flag the call sites. This is deferred (not a
G5 blocker) because the test-side `with_threads(1)` workaround is
zero-cost and the eager-spawn shape is simpler to reason about.

---

## 9. Out of scope (for v1)

- **Row-MT within a tile.** Per HLD §G5, intra-tile parallelism is
  not part of the v1 design. Future row-MT can layer on top of the
  same primitive — each row becomes a `scope.submit(...)` job
  inside a per-tile sub-scope.
- **Affinity / pinning.** No CPU-affinity API. Workers are scheduled
  by the OS.
- **Custom panic hooks.** The pool re-raises panic payloads via
  `std::panic::resume_unwind`; callers can wrap `pool.scope(...)`
  in their own `catch_unwind` to convert panics into errors.
