# `tile_threading` — SAFETY contract

Module-adjacent (per HLD V4 §G5 — not in `notes/`, so refactors that
rename symbols invalidate the doc and a CI doctest catches drift). The
single `unsafe { }` block in this module sits in
[`FrameScope::submit`](scope.rs) where the user closure (with its
captured `'frame` borrows) is type-erased to `Box<dyn FnOnce + Send +
'static>` so it can travel through the
`crossbeam_channel::Sender<Box<dyn FnOnce + Send + 'static>>` job
queue.

This document spells out the soundness chain. Each link references its
enforcement site so a refactor that breaks any link surfaces here at
review time.

## 1. The unsafe site

[`FrameScope::submit`](scope.rs) does, for each job submission:

```rust
let boxed_frame: Box<dyn FnOnce() + Send + 'frame> = Box::new(job_with_signal);
// SAFETY: ...see §2 below.
let boxed_static: Box<dyn FnOnce() + Send + 'static> =
    unsafe { std::mem::transmute(boxed_frame) };
self.pool.job_sender().send(boxed_static as WorkerJob).expect(...);
```

The same function also extends a `&'scope Mutex<...>` to `&'static
Mutex<...>` so the panic-collecting Mutex reference can be captured
in the boxed-as-`'static` closure:

```rust
// SAFETY (panics_ref lifetime extension): ...see §2 link (e).
let panics_ref: &'static Mutex<Vec<(usize, Box<dyn Any + Send + 'static>)>> =
    unsafe { std::mem::transmute(self.panics) };
```

Both transmutes have the same soundness argument: every worker
finishes before [`WorkerPool::scope`](pool.rs) returns. We don't
need the closure to actually be `'static` — we just need it to be
*valid* for the bounded duration of the worker's execution.

## 2. Soundness chain

For the unsafe transmute to be sound, **every reference inside the
type-erased closure must remain valid for the entire duration of the
worker's execution**.

Links:

- **(a) Scope construction.** `WorkerPool::scope<F, R>(&mut self, f: F)
  -> R where F: for<'frame> FnOnce(&FrameScope<'_, 'frame>) -> R`.
  The HRTB `for<'frame>` makes `'frame` a fresh lifetime per call.
  The caller cannot smuggle `&FrameScope` past the closure boundary
  (would require unifying `'frame` with an outer concrete lifetime
  — the HRTB rejects that).

- **(b) Closure-form, not Drop-based.** The caller never sees an
  owning `FrameScope` value. The only access is `&FrameScope` via
  `f`'s argument. `mem::forget(scope)` is therefore structurally
  impossible: there is no scope value to forget. (V3's Drop-based
  shape was unsound under `mem::forget`; V6 K-3 redesigned around
  this.)

- **(c) Unconditional join.** `WorkerPool::scope` runs the user
  closure under `catch_unwind`, then unconditionally drains every
  submitted job before returning. The drain loop is:

  ```rust
  while pending.load(Ordering::SeqCst) > 0 {
      let _outcome = done_rx.recv().expect(...);
      pending.fetch_sub(1, Ordering::SeqCst);
  }
  ```

  Each `FrameScope::submit` does `pending.fetch_add(1)` BEFORE either
  enqueuing the job or reserving it for a caller-participating scope;
  the job wrapper does
  `done_tx.send(WorkerOutcome)` AFTER executing the user closure,
  which causes the next iteration of the drain loop to decrement
  `pending`. By induction, the drain loop only terminates when
  `pending == 0`, which happens only after every submitted job's
  user-closure body has finished.

- **(d) Drain runs on panic too.** The user closure is run under
  `catch_unwind`. If it panics, the drain loop still runs (it sits
  after the `catch_unwind` block, not inside it). If a worker
  panics, the wrapper catches the panic in `catch_unwind` first
  (pushing the payload into the per-scope Mutex), THEN sends the
  done signal — so `pending` is still decremented and the drain
  loop terminates.

  **Liveness invariant (M1).** The worker job wrapper in
  `FrameScope::submit` is structured as a DOUBLE `catch_unwind`:
  the inner catches user-closure panics; the outer catches any
  panic in the wrapper bookkeeping itself (Mutex `lock()` failing
  on poison, `Vec::push` failing on allocator OOM). The
  `done_tx.send(WorkerOutcome)` runs UNCONDITIONALLY after both —
  so `pending` always decrements to 0 within finite time, and the
  drain loop in `WorkerPool::scope` cannot hang. Without this
  double-wrap, a panic in the wrapper bookkeeping would escape to
  `worker_loop`'s outer `catch_unwind` BEFORE the completion
  signal fired, converting a deadlock into a silent infinite
  hang. The pool's `worker_loop` outer `catch_unwind` is still
  belt-and-braces: it prevents a runaway wrapper panic from
  poisoning the worker thread, but the M1 invariant lives in the
  wrapper itself.

- **(e) Mutex reference lifetime.** The per-scope `Mutex<Vec<...>>`
  is a stack local in `WorkerPool::scope`'s frame. Its lifetime
  extension to `'static` in `submit()` is sound because (c) +
  (d) ensure no worker accesses the Mutex after `scope()` returns.
  The Mutex's `Sync` bound makes the `&Mutex<...>` reference
  `Send`-able into the worker closure.

- **(f) Caller participation.** `WorkerPool::new_with_caller`
  reserves one submitted job in a scope-local Mutex. After the scope
  closure returns or unwinds, `WorkerPool::scope` executes that job
  on the caller before entering the same completion drain. The
  reserved closure therefore finishes before its captured `'frame`
  data can expire, under the same unconditional join rule as queued
  worker jobs.

- **(g) Channel ordering.** The job-queue `Sender` and per-scope
  `done_tx` both use `crossbeam_channel`. Sends are observable to
  recipients via cross-thread happens-before edges; the SeqCst
  pending counter cooperates with these to give a total order over
  submit→fetch_add(1)→send and worker→exec→fetch_send(done).

## 3. crossbeam-channel justification

`std::sync::mpsc::Receiver` is `!Sync` (single-consumer): one worker
thread can hold the receiver, but multiple workers cannot share it.
The pool uses MPMC because one job queue feeds many workers (each
competing on `recv`). `crossbeam-channel`'s `Receiver: Sync` permits
the clone-and-share pattern used in `WorkerPool::new`. No equivalent
exists in `std`.

## 4. Why two transmutes, not one

The closure-erasure transmute and the Mutex-reference-erasure
transmute both rely on the SAME soundness chain (§2 above). They are
spelled out as two transmutes only because Rust's lifetime elision
needs them at distinct points (the Mutex lives on the scope's stack
frame; the user closure may borrow from the OUTER caller's stack
frame). Both lifetimes are bounded by (c)+(d)'s unconditional join.

## 5. What this contract does NOT cover

- **Worker-side data races.** This contract makes the job dispatch
  itself safe. It does NOT say that the *user closure's accesses*
  are race-free. The decoder-side caller (G5 integration) is
  responsible for handing out disjoint write regions to each
  worker (see HLD V4 §G5 option (a) `split_into_tile_views` or
  fallback (b) UnsafeCell+TileSlot).

- **Worker panic propagation across scopes.** Panics that happen
  AFTER `scope()` returns (e.g., a worker that's still spinning on
  the channel because the pool hasn't been dropped yet) are
  irrelevant: there's nothing to propagate, and Drop swallows them
  silently. This is identical to `std::thread::scope`'s behaviour.

- **Re-entrancy.** Calling `pool.scope()` from inside a worker job
  would deadlock (pool has unique borrow). The `&mut self` on
  `scope()` makes this a compile error.

## 6. Drift detection

A CI doctest (`#[cfg(doctest)]`) should be added once the suite is
in place. For now: this document lives at
`crates/rvp9-core/src/tile_threading/SAFETY.md`. If the symbols
named here (`WorkerPool::scope`, `FrameScope::submit`, the
`pending` / `done_rx` / `panics` triple, the unsafe transmutes) are
renamed or moved, this document MUST be updated alongside.

## 7. `LfWorkerPool` — Phase 1 separate-pool LF overlap

`LfWorkerPool` (in [`lf_pool.rs`](lf_pool.rs)) is a structurally
parallel, purpose-specialised pool used by the Phase 1 separate-pool
LF overlap (HLD `2026.05.27 - HLD - Separate LF thread pool.md`). It
mirrors libvpx's `pbi->lf_worker` non-row_mt topology — a long-lived
worker thread that runs the column-wavefront LF kernel **concurrently
with** the main decode worker pool, gated on the per-SB-row
`decode_progress` atomic vector published by tile workers.

### 7.1 The unsafe site

[`LfWorkerPool::dispatch`](lf_pool.rs) has the same closure-erasure
transmute as `FrameScope::submit`:

```rust
let boxed_frame: Box<dyn FnOnce() + Send + 'frame> = Box::new(job_with_signal);
let boxed_static: Box<dyn FnOnce() + Send + 'static> =
    unsafe { std::mem::transmute(boxed_frame) };
```

### 7.2 Soundness chain (`LfBatchHandle::wait` joins the borrow)

The chain mirrors §2 above with one shape change: instead of a
blocking scope-API (`scope()` joins before returning), `dispatch`
returns an `LfBatchHandle<'pool, 'frame>` whose `wait()` is the
temporal-exclusion edge:

- **(a) Lifetime parameterisation.** `LfBatchHandle` carries
  `PhantomData<&'frame ()>` so the borrow checker treats the handle
  as borrowing `'frame`. The caller cannot drop the borrowed data
  before consuming the handle.

- **(b) `wait()` joins.** The wrapper around the user closure does
  `done_tx.send(LfOutcome)` UNCONDITIONALLY after running the closure
  (same liveness invariant M1 as `FrameScope::submit`). `wait()`
  blocks on `done_rx.recv()`. After `wait()` returns, no worker
  thread is accessing the closure's `'frame`-borrowed data.

- **(c) `Drop` is the fallback.** If the caller does not call
  `wait()` explicitly, `LfBatchHandle::Drop` blocks on `done_rx.recv`
  before returning — a best-effort liveness path. The structural
  soundness comes from the `'frame` lifetime in the handle's type:
  the borrow checker disallows dropping the borrowed data before
  dropping the handle.

- **(d) `mem::forget` defense — documented contract on `unsafe fn
  dispatch`.** `LfBatchHandle: !Copy`, but `mem::forget(handle)` is
  type-correct: it consumes the handle (ending its `'frame` borrow)
  WITHOUT running `Drop`, which would let the LF worker continue
  raw-pointer access to per-frame state past the borrow's end —
  undefined behaviour. The borrow-checker does NOT structurally
  prevent this; the previous claim that the `'frame` PhantomData
  "structurally prevents `mem::forget`" was incorrect.

  Instead, `LfWorkerPool::dispatch` is marked **`unsafe fn`**, and
  the documented safety contract obliges every caller to wrap the
  call in `unsafe { ... }` and uphold the no-`mem::forget`
  invariant. Best-effort defence-in-depth: the handle's own `Drop`
  blocks on `done_rx.recv` so the natural end-of-scope path is
  sound. The two production call sites
  (`crates/rvp9-decoder/src/lib.rs` intra and inter LF-overlap
  dispatches) bind the handle to a named local and consume it via
  `handle.wait()` after the decode scope; neither path forgets it.
  In-tree tests follow the same pattern.

- **(e) Cross-pool ordering.** Decode workers and LF workers live in
  **physically separate** OS thread pools (`WorkerPool` vs
  `LfWorkerPool`). They share access to:
  - The frame buffer via `SharedFrame` raw pointers (decode writes
    pixels; LF reads-then-writes pixels).
  - The `decode_progress: &[AtomicU32]` slice (decode writes via
    `fetch_add(1, Release)` per SB row; LF reads via
    `load(Acquire)`).
  - The `frame_mi_grid: &[Option<ModeInfo>]` (decode writes via
    `TileMiView`; LF reads via `RowMaskInputs.mi_grid`).

  The **temporal partitioning** that makes the shared access sound:
  - **Invariant L1 (HLD §4.3):** LF of row `r` may begin only after
    `decode_progress[(r+lf_delay).min(sb_rows-1)] >= num_tile_cols`,
    where `lf_delay = 1`. Each tile worker bumps
    `decode_progress[r]` with `Release` after finishing all writes
    to row `r`'s pixels and mi-grid cells. LF's `Acquire` load
    of the same atomic happens-before its subsequent reads. The
    Acquire/Release pair is thread-set-agnostic in the Rust memory
    model — cross-pool synchronisation is identical to in-pool
    synchronisation.
  - **Intra-frame VP9 spec:** intra-prediction at row `r` reads
    neighbours from rows ≤ `r`. With `lf_delay = 1`, LF of row `r-1`
    starts only after every tile completed row `r`, which is after
    every intra-pred read of row `r-1`'s pixels. No read-vs-write
    race.

### 7.3 `SharedFrame` semantics — unchanged

`LfWorkerPool` does NOT modify `SharedFrame`'s ownership semantics
(HLD §6 DO-NOT list #5; preserved from Lever B v2). Cross-pool
aliasing is sound because of (e) above, not because `SharedFrame`
relaxes any invariant.

### 7.4 What this contract does NOT cover (LfWorkerPool)

- **Borrow ordering inside the LF worker.** The LF state's `'frame`
  references (decode_progress, cur_sb_col, frame_mi_grid, levels,
  info, SharedFrame) all share one `'frame` lifetime. The handle
  parameterises over `'frame` and `wait()`s before any of them go
  out of scope. Caller is responsible for keeping all state alive
  through the matching `wait()`.

- **Reentrancy.** `LfWorkerPool` is not designed for nested
  dispatches (the single long-lived thread serialises them). A future
  N_LF > 1 patch will need to revisit the dispatch protocol.
