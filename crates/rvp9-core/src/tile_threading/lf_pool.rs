//! Long-lived loop-filter worker pool (Phase 1, separate-pool LF overlap).
//!
//! Mirrors libvpx's `pbi->lf_worker` non-row_mt overlap topology
//! (`third_party/libvpx/vp9/decoder/vp9_decodeframe.c:2027-2151`): a
//! dedicated long-lived worker thread that runs the loop-filter kernel
//! while the main decode worker pool is still chewing tile-decode work.
//!
//! ## API shape: long-lived dispatch + handle
//!
//! Per the signed HLD (`wrk_docs/2026.05.27 - HLD - Separate LF thread
//! pool.md` §3.3 option (b) / §11 condition 1):
//!
//! ```ignore
//! // SAFETY: handle.wait() (or Drop) joins before the borrowed state
//! // expires; we never `mem::forget` the handle. See dispatch() docs.
//! let handle = unsafe { lf_pool.dispatch(|| run_lf_jobs()) };
//! decode_pool.scope(|scope| { /* decode tiles */ });  // runs in parallel
//! handle.wait();                                       // join the LF thread
//! ```
//!
//! Unlike [`super::WorkerPool::scope`] (whose `scope()` method is a
//! blocking call), [`LfWorkerPool::dispatch`] returns a handle the
//! caller can park on later. This lets the main thread run the decode
//! pool's scope concurrently with the LF pool's work.
//!
//! `dispatch` is `unsafe` because it requires the caller to honour a
//! no-`mem::forget` invariant on the returned handle: the handle's
//! `Drop` (or `wait()`) is the temporal-exclusion edge that
//! synchronises the LF worker's raw-pointer access to per-frame state
//! with the frame's deconstruction. `mem::forget` would skip `Drop`
//! and let the worker access freed state — UB. See
//! [`LfWorkerPool::dispatch`]'s `# Safety` block for the full
//! contract.
//!
//! ## Phase 1: N_LF = 1
//!
//! The team-lead binding condition (HLD §11 condition 1) is `N_LF = 1`
//! to match libvpx exactly. This module therefore intentionally exposes
//! only a *single-worker* pool. Generalising to N > 1 is a Phase 2
//! experiment requiring a separate sign-off; the API is sized so the
//! later extension is a closed-set refactor (the dispatch closure runs
//! `N_LF` "logical worker" submissions today; with N_LF=1 they are all
//! served by the one OS thread, but a future N_LF=2 patch only needs to
//! spawn extra workers and stripe submissions across them).
//!
//! ## SAFETY contract
//!
//! The single `unsafe { mem::transmute }` block lives in
//! [`LfWorkerPool::dispatch`]. The job closure may borrow data from the
//! caller's stack frame (`'frame`). It is type-erased to `'static` so
//! the boxed job can travel through `crossbeam_channel`. The transmute
//! is sound because [`LfBatchHandle::wait`] joins the LF worker before
//! returning to the caller, and the handle's `'frame` lifetime parameter
//! ties the borrow to the caller's stack frame — see
//! `crates/rvp9-core/src/tile_threading/SAFETY.md` for the full chain.

use crossbeam_channel::{Receiver, Sender, unbounded};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread::{self, JoinHandle};

/// Type-erased job sent across the LF dispatch channel. The closure
/// captures the per-frame state by raw pointer (`SharedFrame`,
/// `&[AtomicU32]`, `&[AtomicI32]`); the temporal-exclusion edge that
/// makes those captures sound is the matching
/// [`LfBatchHandle::wait`].
type LfJob = Box<dyn FnOnce() + Send + 'static>;

/// Message the dispatcher sends to the long-lived LF worker thread.
enum LfControl {
    /// Run one batch closure. The closure is responsible for signalling
    /// completion via the per-dispatch `done_tx` it captures.
    Dispatch(LfJob),
    /// Pool is being dropped — worker exits its loop.
    Shutdown,
}

/// Long-lived loop-filter worker pool. Construct once per `Decoder`;
/// joined on [`Drop`].
///
/// Phase 1 (HLD §11 condition 1): the pool spawns **exactly one**
/// worker thread. Future N_LF > 1 is a Phase 2 experiment.
///
pub struct LfWorkerPool {
    /// Worker thread handles. Joined in [`Drop`].
    workers: Vec<JoinHandle<()>>,
    /// Control sender. `Option` so [`Drop`] can drop it (closing the
    /// channel; workers' `recv()` returns `Err` and they exit).
    control_tx: Option<Sender<LfControl>>,
    /// Number of LF workers (for `num_workers` plumbing — Phase 1 = 1).
    num_workers: NonZeroUsize,
}

impl std::fmt::Debug for LfWorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LfWorkerPool")
            .field("num_workers", &self.num_workers.get())
            .field("alive", &self.control_tx.is_some())
            .finish()
    }
}

impl LfWorkerPool {
    /// Construct an LF pool with `n` long-lived worker threads.
    ///
    /// Phase 1 callers MUST pass `n = 1` (HLD §11 condition 1). The
    /// `NonZeroUsize` shape leaves room for a future N_LF > 1 patch
    /// without an API break.
    /// # Panics
    ///
    /// Panics if the operating system refuses to create a worker thread.
    /// User-controlled construction should use [`Self::try_new`] instead.
    #[must_use]
    pub fn new(n: NonZeroUsize) -> Self {
        Self::try_new(n).expect("rvp9 LF worker thread spawn failed")
    }

    /// Construct an LF pool, returning worker thread spawn failures.
    ///
    /// # Errors
    ///
    /// Returns an error if handle storage cannot be reserved or the operating
    /// system refuses to spawn a worker.
    pub fn try_new(n: NonZeroUsize) -> std::io::Result<Self> {
        Self::try_new_with_spawner(n, |worker_id, rx| {
            thread::Builder::new()
                .name(format!("rvp9-lf-worker-{worker_id}"))
                .spawn(move || lf_worker_loop(rx))
        })
    }

    fn try_new_with_spawner<F>(n: NonZeroUsize, mut spawn_worker: F) -> std::io::Result<Self>
    where
        F: FnMut(usize, Receiver<LfControl>) -> std::io::Result<JoinHandle<()>>,
    {
        let (control_tx, control_rx) = unbounded::<LfControl>();
        let mut workers = Vec::new();
        workers.try_reserve_exact(n.get()).map_err(|error| {
            std::io::Error::other(format!("reserving LF worker handles: {error}"))
        })?;
        for worker_id in 0..n.get() {
            let rx: Receiver<LfControl> = control_rx.clone();
            match spawn_worker(worker_id, rx) {
                Ok(handle) => workers.push(handle),
                Err(error) => {
                    drop(control_tx);
                    drop(control_rx);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(error);
                }
            }
        }
        // Drop the local Receiver so the channel actually closes when
        // `control_tx` is dropped at pool teardown.
        drop(control_rx);
        Ok(Self {
            workers,
            control_tx: Some(control_tx),
            num_workers: n,
        })
    }

    /// Worker count (Phase 1 = 1, by HLD §11 condition 1).
    #[inline]
    #[must_use]
    pub fn num_workers(&self) -> NonZeroUsize {
        self.num_workers
    }

    /// Dispatch the LF batch closure to the long-lived worker thread
    /// and return a handle the caller can later `wait()` on.
    ///
    /// The closure may borrow data from the caller's stack frame (the
    /// `'frame` HRTB). The returned [`LfBatchHandle`] carries the same
    /// `'frame` lifetime so the Rust borrow checker prevents the caller
    /// from dropping the borrowed data before `wait()`.
    ///
    /// # Safety
    ///
    /// The caller **MUST NOT** `mem::forget` the returned
    /// [`LfBatchHandle`]. The handle's `Drop` (or explicit
    /// [`LfBatchHandle::wait`]) is the temporal-exclusion edge that
    /// synchronises the LF worker's raw-pointer access to per-frame
    /// state with the frame's deconstruction. `mem::forget` skips
    /// `Drop` and would let the worker continue accessing freed
    /// state — undefined behaviour.
    ///
    /// Callers that follow the documented
    /// `let h = unsafe { pool.dispatch(...) }; ...; h.wait();`
    /// pattern (or simply let the handle drop at end-of-scope) are
    /// sound. See `crates/rvp9-core/src/tile_threading/SAFETY.md` §7
    /// for the full chain.
    ///
    /// # Soundness (impl detail)
    ///
    /// The transmute below extends `'frame` to `'static` so the boxed
    /// closure can travel through the `Sender<LfJob>` channel. This is
    /// sound *provided the caller upholds the above contract*, because
    /// [`LfBatchHandle::wait`] (or its `Drop`) then joins the worker
    /// before returning — see SAFETY.md and the per-line comments
    /// below.
    pub unsafe fn dispatch<'pool, 'frame, F>(&'pool self, job: F) -> LfBatchHandle<'pool, 'frame>
    where
        F: FnOnce() + Send + 'frame,
    {
        let (done_tx, done_rx) = unbounded::<LfOutcome>();

        // Wrap the user closure in a panic-catching, completion-
        // signalling shell — same liveness invariant as
        // `tile_threading::scope::FrameScope::submit` (M1 fix): the
        // `done_tx.send` MUST run unconditionally so `wait()` cannot
        // hang on a panicking job.
        //
        // **Ordering note** (Stacked/Tree Borrows requirement):
        // `done_tx.send` MUST be the LAST thing the wrapper does, AND
        // the user closure (`job`) MUST be fully dropped BEFORE the
        // send fires. Otherwise the main thread's `wait()` could
        // return — and the caller could start invalidating
        // `'frame`-borrowed state — while the worker still holds a
        // protected reference to that state via the not-yet-dropped
        // user closure box.
        //
        // What enforces the drop-before-send ordering: **`FnOnce`
        // semantics inside `catch_unwind`**.
        // `catch_unwind(AssertUnwindSafe(job))` moves `job` by value
        // into its inner `FnOnce::call_once` frame; that frame
        // executes the body and then drops `job`'s captures before
        // `catch_unwind` returns. The local binding `result` is just
        // syntactic — by the time control reaches the `done_tx.send`
        // line, the captures are already gone regardless of whether
        // we bind or inline the `match`.
        let job_with_signal = move || {
            // Run + drop user closure (`job` moved into catch_unwind,
            // consumed by the inner FnOnce call; the call frame drops
            // the captures before returning). The result's Err
            // variant carries the panic payload as
            // `Box<dyn Any + Send + 'static>`, which does NOT name
            // `'frame`.
            let result = catch_unwind(AssertUnwindSafe(job));
            // `result` no longer holds any `'frame` borrow at this
            // point — see the FnOnce/catch_unwind ordering note
            // above.
            let outcome = match result {
                Ok(()) => LfOutcome::Ok,
                Err(payload) => LfOutcome::Panic(payload),
            };
            // Final signal. Best-effort: if the receiver is gone the
            // handle was dropped without wait — we still want the
            // worker thread to return cleanly so it can pick up the
            // next dispatch.
            let _ = done_tx.send(outcome);
        };

        let boxed_frame: Box<dyn FnOnce() + Send + 'frame> = Box::new(job_with_signal);
        // SAFETY:
        // (a) The user closure `F: FnOnce() + Send + 'frame` may borrow
        //     data from the caller's stack frame (per-frame
        //     `ColumnWavefrontLfState`, the `decode_progress` atomic
        //     slice, etc.). The 'frame lifetime expresses that borrow.
        // (b) The returned `LfBatchHandle<'pool, 'frame>` carries
        //     `'frame` in its type. The caller cannot drop the borrowed
        //     data before dropping (or `wait()`-ing on) the handle:
        //     the handle holds a `PhantomData<&'frame ()>`, so any
        //     code that invalidates the `'frame` borrow before the
        //     handle is consumed is a borrow-checker error.
        // (c) `LfBatchHandle::wait` blocks on `done_rx.recv()`, which
        //     returns only after the worker's `job_with_signal`
        //     wrapper has finished executing the user closure
        //     (unconditional `done_tx.send` after `catch_unwind`). So
        //     by the time `wait()` returns, no worker is still
        //     accessing the closure's 'frame-borrowed data.
        // (d) `mem::forget` defense (this is the `unsafe fn` contract).
        //     `mem::forget(handle)` is type-correct: the handle's
        //     `'frame` PhantomData only ties the *handle value* to
        //     `'frame`, and `mem::forget` consumes the handle (ending
        //     the borrow) WITHOUT running `Drop` — letting the worker
        //     continue raw-pointer access to per-frame state past the
        //     borrow's end. The borrow checker does NOT prevent this.
        //     The protection is the **documented contract** on
        //     `unsafe fn dispatch`: callers wrap the call in
        //     `unsafe { }` and uphold the no-`mem::forget`
        //     invariant. Best-effort defence-in-depth: the handle's
        //     own `Drop` blocks on `done_rx.recv` so the natural
        //     end-of-scope path is sound; `mem::forget` is the only
        //     unsound escape and is explicitly forbidden by the
        //     contract.
        // (e) The closure is `Send` because every capture
        //     (`SharedFrame` raw pointers, `&[AtomicU32]` references,
        //     channel `Sender` clones) is `Send`.
        let boxed_static: Box<dyn FnOnce() + Send + 'static> =
            unsafe { std::mem::transmute(boxed_frame) };

        self.control_tx
            .as_ref()
            .expect("LfWorkerPool::dispatch called after Drop began")
            .send(LfControl::Dispatch(boxed_static))
            .expect("rvp9 LfWorkerPool: control channel closed mid-dispatch");

        LfBatchHandle {
            done_rx: Some(done_rx),
            _pool: PhantomData,
            _frame: PhantomData,
        }
    }
}

impl Drop for LfWorkerPool {
    fn drop(&mut self) {
        if let Some(tx) = self.control_tx.take() {
            // Send explicit shutdown messages — one per worker — so
            // workers exit cleanly even if the channel-close path is
            // racy on some platforms. Closing `tx` afterwards causes
            // any remaining workers to see `Err(_)` from `recv` and
            // exit as a belt-and-braces second path.
            for _ in 0..self.workers.len() {
                let _ = tx.send(LfControl::Shutdown);
            }
            drop(tx);
        }
        for handle in self.workers.drain(..) {
            // We're already in Drop; can't propagate. The pool's main
            // panic-surfacing path is the per-dispatch `done` channel,
            // not Drop.
            let _ = handle.join();
        }
    }
}

/// Handle to a pending LF batch. Block via [`wait`](Self::wait).
///
/// The `'frame` lifetime ties the handle to the caller's stack frame —
/// the borrow checker prevents the caller from dropping the
/// borrowed-by-the-closure data before consuming the handle.
///
/// **Drop semantics:** dropping the handle without calling `wait()`
/// blocks on the worker's completion signal anyway (the temporal-
/// exclusion edge for the unsafe `dispatch` contract — see SAFETY
/// chain note (d) in [`LfWorkerPool::dispatch`]). The intended usage
/// is always explicit `handle.wait()`; `mem::forget(handle)` is
/// forbidden by the `dispatch` safety contract because it would skip
/// this `Drop`.
#[must_use = "LfBatchHandle must be wait()-ed on (or dropped) so the LF worker completes before the borrowed frame state expires"]
pub struct LfBatchHandle<'pool, 'frame> {
    /// `Option` so `wait()` can take the receiver out (and Drop can
    /// check whether `wait()` already ran).
    done_rx: Option<Receiver<LfOutcome>>,
    /// Tie the handle to the pool — the pool must outlive the handle
    /// (otherwise its worker threads are gone and we'd deadlock on
    /// `done_rx`).
    _pool: PhantomData<&'pool LfWorkerPool>,
    /// Tie the handle to the caller's `'frame` borrow — see SAFETY
    /// chain (b) in [`LfWorkerPool::dispatch`].
    _frame: PhantomData<&'frame ()>,
}

impl<'pool, 'frame> LfBatchHandle<'pool, 'frame> {
    /// Block until the dispatched LF batch finishes. Propagates a
    /// worker panic by re-raising it on the caller's thread.
    pub fn wait(mut self) {
        let rx = self
            .done_rx
            .take()
            .expect("LfBatchHandle::wait called twice (Option already taken)");
        match rx.recv() {
            Ok(LfOutcome::Ok) => {}
            Ok(LfOutcome::Panic(payload)) => {
                std::panic::resume_unwind(payload);
            }
            Err(_) => {
                // Worker thread died without signalling — shouldn't
                // happen because the wrapper sends unconditionally.
                // Treat as fatal logic error.
                panic!("rvp9 LfWorkerPool: done channel closed before completion signal");
            }
        }
    }
}

impl<'pool, 'frame> Drop for LfBatchHandle<'pool, 'frame> {
    fn drop(&mut self) {
        if let Some(rx) = self.done_rx.take() {
            // Caller didn't call wait() — block here as a best-effort
            // liveness path so the worker's access to `'frame` data
            // completes before the handle's lifetime ends. Swallow any
            // panic payload (we cannot resume_unwind from Drop).
            let _ = rx.recv();
        }
    }
}

/// Outcome of one dispatched LF batch.
enum LfOutcome {
    Ok,
    Panic(Box<dyn std::any::Any + Send + 'static>),
}

/// Worker thread entry point: pull control messages from the channel
/// until either an explicit `Shutdown` or the channel closes.
fn lf_worker_loop(rx: Receiver<LfControl>) {
    while let Ok(msg) = rx.recv() {
        match msg {
            LfControl::Dispatch(job) => {
                // The job's own `job_with_signal` wrapper already
                // catches panics. This outer catch_unwind is
                // belt-and-braces in case the wrapper panics during
                // bookkeeping (allocator OOM).
                let _ = catch_unwind(AssertUnwindSafe(job));
            }
            LfControl::Shutdown => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    #[test]
    fn spawn_failure_is_returned() {
        let error = LfWorkerPool::try_new_with_spawner(nz(2), |worker_id, receiver| {
            if worker_id == 0 {
                Ok(thread::spawn(move || lf_worker_loop(receiver)))
            } else {
                Err(std::io::Error::other("injected LF spawn failure"))
            }
        })
        .expect_err("spawn failure must be returned");

        assert_eq!(error.kind(), std::io::ErrorKind::Other);
        assert_eq!(error.to_string(), "injected LF spawn failure");
    }

    /// Smoke test: dispatch + wait runs the closure exactly once.
    #[test]
    fn dispatch_runs_once() {
        let pool = LfWorkerPool::new(nz(1));
        let counter = AtomicUsize::new(0);
        let counter_ref = &counter;
        // SAFETY: handle is wait()-ed below; never mem::forget'd —
        // upholds the LfWorkerPool::dispatch no-forget contract.
        let handle = unsafe {
            pool.dispatch(|| {
                counter_ref.fetch_add(1, Ordering::SeqCst);
            })
        };
        handle.wait();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    /// Sequential dispatches reuse the same long-lived thread.
    #[test]
    fn dispatch_reusable_across_calls() {
        let pool = LfWorkerPool::new(nz(1));
        let counter = AtomicUsize::new(0);
        let counter_ref = &counter;
        for _ in 0..4 {
            // SAFETY: handle is wait()-ed immediately; never
            // mem::forget'd — upholds the dispatch no-forget contract.
            unsafe {
                pool.dispatch(|| {
                    counter_ref.fetch_add(1, Ordering::SeqCst);
                })
            }
            .wait();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    /// The dispatched closure may borrow `'frame` data via shared
    /// references — exercises the unsafe transmute's soundness.
    ///
    /// **Why `&AtomicUsize` and not `&Mutex<Vec>`** (which would more
    /// directly mirror `FrameScope::submit`'s differential test):
    /// Stacked Borrows tracks `&Mutex` references as "strongly
    /// protected" when they are passed as function arguments. With
    /// the closure consumed inside a `catch_unwind` running on the
    /// worker thread, SB can't see that the captured borrow is
    /// released before the main thread proceeds past `wait()` — even
    /// though Tree Borrows accepts the same pattern. Using
    /// `&AtomicUsize` instead avoids the SB tracking entirely
    /// (atomics aren't borrow-tracked) while still exercising the
    /// `'frame`-lifetime transmute we want to assert sound. The
    /// production path uses `SharedFrame` raw pointers (also untracked
    /// by SB), so this matches actual usage.
    #[test]
    fn dispatched_closure_may_borrow_frame_data() {
        let pool = LfWorkerPool::new(nz(1));
        let counter = AtomicUsize::new(0);
        let counter_ref = &counter;
        // SAFETY: handle is wait()-ed below; never mem::forget'd —
        // upholds the dispatch no-forget contract.
        let handle = unsafe {
            pool.dispatch(move || {
                for i in 0..8usize {
                    counter_ref.fetch_add(i, Ordering::SeqCst);
                }
            })
        };
        handle.wait();
        // Sum of 0..8 = 28
        assert_eq!(counter.load(Ordering::SeqCst), 28);
    }

    /// A panic in the dispatched closure propagates on `wait()`.
    #[test]
    fn dispatched_panic_propagates_on_wait() {
        let pool = LfWorkerPool::new(nz(1));
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: handle is wait()-ed below; never mem::forget'd —
            // upholds the dispatch no-forget contract.
            let handle = unsafe { pool.dispatch(|| panic!("boom")) };
            handle.wait();
        }));
        let payload = result.expect_err("expected a panic to propagate");
        let msg = payload
            .downcast_ref::<&'static str>()
            .copied()
            .unwrap_or_default();
        assert_eq!(msg, "boom");
    }

    /// Dropping the handle without `wait()` still blocks so the
    /// borrowed data outlives the worker access. This is a liveness
    /// (not soundness) check — soundness comes from the borrow
    /// checker tying `'frame` to the handle.
    #[test]
    fn drop_without_wait_blocks_until_worker_done() {
        let pool = LfWorkerPool::new(nz(1));
        let counter = AtomicUsize::new(0);
        let counter_ref = &counter;
        {
            // SAFETY: handle is dropped (not mem::forget'd) at the end
            // of the inner block — upholds the dispatch no-forget
            // contract; this test exercises the Drop fallback path.
            let _handle = unsafe {
                pool.dispatch(|| {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    counter_ref.fetch_add(1, Ordering::SeqCst);
                })
            };
            // _handle drops here; Drop blocks on recv().
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
