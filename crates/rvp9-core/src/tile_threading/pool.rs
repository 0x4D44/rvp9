//! Worker pool — long-lived OS threads driven by a `crossbeam_channel`
//! job queue. See [`mod@super`] for the API overview.

use crossbeam_channel::{Receiver, Sender, unbounded};
use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

use super::scope::FrameScope;

/// Type-erased job sent to a worker. The closure captures everything
/// it needs, including a per-scope completion signal (a `Sender` of
/// `thread::Result<()>`).
pub(crate) type WorkerJob = Box<dyn FnOnce() + Send + 'static>;

/// Long-lived worker pool. Construct once per codec instance via
/// [`WorkerPool::new`]; joined on drop.
///
/// `N = 1` is the conformance-deterministic single-threaded knob: it
/// still spawns the channel infrastructure (and one worker thread) so
/// the API shape is identical to the multi-threaded path, but the
/// decoder convention (HLD V6 §G5 V5 "inline-worker pattern") runs all
/// tiles on the caller thread anyway and never submits work — see the
/// decoder-side `decode_tile_job` dispatcher.
///
/// The pool itself is `Send + Sync`: workers consume from an MPMC
/// receiver, so the `Sender` clone in each [`FrameScope`] does not
/// require `&mut self`. Construction-time `&mut WorkerPool` borrow
/// in `scope()` is structural — it bounds `'pool` and forbids
/// nested scopes on the same pool (which would tangle pending
/// counters).
pub struct WorkerPool {
    /// Worker thread handles. Joined in [`Drop`].
    workers: Vec<JoinHandle<()>>,
    /// Job sender. `Option` so [`Drop`] can drop it (closing the
    /// channel; workers' `recv()` returns `Err` and they exit).
    job_tx: Option<Sender<WorkerJob>>,
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("num_workers", &self.workers.len())
            .field("alive", &self.job_tx.is_some())
            .finish()
    }
}

impl WorkerPool {
    /// Construct a pool with `n` worker threads.
    ///
    /// Workers spin in `recv()` until [`Drop`] closes the channel.
    /// Job queue uses an unbounded channel — flow control happens
    /// per-frame via [`FrameScope`]'s pending counter, not at the
    /// queue boundary.
    // TODO(G5.1): consider lazy worker spawning — workers should be
    // created on first `scope()` invocation, not on `WorkerPool::new()`.
    // Today's eager spawn means each `WorkerPool::new(n)` spawns
    // `n` OS threads even when the caller never submits work, which
    // amplifies test/fuzz cost (every `Decoder::new()` defaults to
    // `available_parallelism()` threads). See
    // `notes/lld/core-tile-threading.md` § "Performance/cost notes".
    /// # Panics
    ///
    /// Panics if the operating system refuses to create a worker thread.
    /// User-controlled thread counts should use [`Self::try_new`] instead.
    #[must_use]
    pub fn new(n: NonZeroUsize) -> Self {
        Self::try_new(n).expect("rvp9 worker thread spawn failed")
    }

    /// Construct a pool with `n` worker threads, returning spawn failures.
    ///
    /// If a later worker fails to spawn, already-created workers are shut
    /// down and joined before the error is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if handle storage cannot be reserved or the operating
    /// system refuses to spawn a worker.
    pub fn try_new(n: NonZeroUsize) -> std::io::Result<Self> {
        Self::try_new_with_spawner(n, |worker_id, rx| {
            thread::Builder::new()
                .name(format!("rvp9-worker-{worker_id}"))
                .spawn(move || worker_loop(rx))
        })
    }

    fn try_new_with_spawner<F>(n: NonZeroUsize, mut spawn_worker: F) -> std::io::Result<Self>
    where
        F: FnMut(usize, Receiver<WorkerJob>) -> std::io::Result<JoinHandle<()>>,
    {
        let n = n.get();
        let (job_tx, job_rx) = unbounded::<WorkerJob>();
        let mut workers = Vec::new();
        workers
            .try_reserve_exact(n)
            .map_err(|error| std::io::Error::other(format!("reserving worker handles: {error}")))?;
        for worker_id in 0..n {
            let rx: Receiver<WorkerJob> = job_rx.clone();
            match spawn_worker(worker_id, rx) {
                Ok(handle) => workers.push(handle),
                Err(error) => {
                    drop(job_tx);
                    drop(job_rx);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(error);
                }
            }
        }
        // Drop the local Receiver so once `job_tx` is dropped at
        // pool teardown, the channel actually closes (workers see
        // `Err(_)` from `recv` and exit).
        drop(job_rx);
        Ok(Self {
            workers,
            job_tx: Some(job_tx),
        })
    }

    /// Open a per-frame scope and run `f` with mutable access to it.
    ///
    /// Blocks until every job submitted from inside `f` has completed
    /// (regardless of whether `f` returned normally or unwound).
    ///
    /// **Panic protocol** (HLD V6 §G5 two-failure-surface model):
    ///
    /// 1. If `f` itself panics, all submitted jobs are still joined,
    ///    then `f`'s panic payload is re-raised. Worker panics
    ///    during the same scope are intentionally dropped on the
    ///    floor — the closure failure is primary.
    /// 2. If `f` returns normally but any submitted job panicked, the
    ///    FIRST panic in submission order is re-raised.
    /// 3. If `f` returns normally and no job panicked, `f`'s return
    ///    value is returned.
    ///
    /// The `'pool` borrow makes nested scopes on the same pool a
    /// compile error — the inner `scope()` would need a second
    /// `&mut self`, which the outer scope still holds.
    pub fn scope<F, R>(&mut self, f: F) -> R
    where
        F: for<'frame> FnOnce(&FrameScope<'_, 'frame>) -> R,
    {
        // Per-scope channel: workers signal completion (and panic
        // payload, if any) here. Bounded(0) would deadlock at submit
        // time (would block the worker until the main thread receives);
        // unbounded is safe because the queue length is bounded by
        // pending count anyway.
        //
        // The Mutex around the panic-payload Vec gives us O(1)
        // ordered push per worker. Worker order matches the SeqCst
        // pending-counter ordering — the first job to fetch_add(1)
        // gets the lowest submission index, the first to finish may
        // be a different one, but ordering is by submission index
        // not completion order. See the worker_id assignment in
        // [`FrameScope::submit`].
        let (done_tx, done_rx) = unbounded::<WorkerOutcome>();

        // Build the scope. Borrowing `self` (the pool) gives the
        // scope a way to access the job channel. Pending counter +
        // done channel are scope-local.
        let pending = AtomicUsize::new(0);
        let panics: Mutex<Vec<(usize, Box<dyn std::any::Any + Send + 'static>)>> =
            Mutex::new(Vec::new());
        let scope = FrameScope::new(self, &pending, &done_tx, &panics);

        // Run the closure under catch_unwind so we can join cleanly
        // even on panic.
        let closure_result = catch_unwind(AssertUnwindSafe(|| f(&scope)));

        // Drop the local `done_tx` clone so once all workers have
        // their per-job senders dropped (post-job), the channel
        // empties. Actually — workers don't own done_tx directly;
        // each job closure captures a clone. The pending counter is
        // what we rely on.
        drop(done_tx);

        // Drain the done channel until pending == 0. Each WorkerOutcome
        // signals one completed job (success or panic). We DON'T
        // re-raise here — we collect into the panics Vec and decide
        // priority after.
        while pending.load(Ordering::SeqCst) > 0 {
            match done_rx.recv() {
                Ok(_outcome) => {
                    // Outcome already pushed by the worker into the
                    // panics Vec if it was a panic. Decrement the
                    // pending counter.
                    pending.fetch_sub(1, Ordering::SeqCst);
                }
                Err(_) => {
                    // Channel disconnected with pending > 0 — only
                    // possible if a worker thread died without
                    // signalling, which can't happen because our
                    // worker loop always sends a completion. Treat
                    // as a fatal logic error.
                    panic!(
                        "rvp9 WorkerPool: done channel closed with pending jobs ({} left)",
                        pending.load(Ordering::SeqCst)
                    );
                }
            }
        }

        // Decide which payload (if any) to re-raise. After the M1
        // liveness fix in `FrameScope::submit`, the wrapper holds
        // the panics-vec lock only inside a `catch_unwind`, so this
        // mutex cannot be poisoned by an in-flight unwind. Reaching
        // this `expect` would mean the wrapper itself panicked while
        // holding the lock (e.g. allocator OOM during `Vec::push`),
        // which is a fatal-environment scenario where panicking
        // here is the correct response.
        let collected_panics = panics
            .into_inner()
            .expect("panics mutex poisoned — allocator OOM inside scope wrapper");
        match closure_result {
            Err(closure_panic) => {
                // Closure panic wins. Worker panics dropped on the floor.
                resume_unwind(closure_panic);
            }
            Ok(r) => {
                if let Some((_idx, payload)) = first_worker_panic(collected_panics) {
                    resume_unwind(payload);
                }
                r
            }
        }
    }

    /// Internal: clone the job sender. Used by [`FrameScope::submit`]
    /// to enqueue work.
    pub(crate) fn job_sender(&self) -> &Sender<WorkerJob> {
        self.job_tx
            .as_ref()
            .expect("WorkerPool::scope called after Drop began")
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        // 1. Drop the job sender. Workers' `recv()` returns Err and
        //    they exit their loops.
        drop(self.job_tx.take());
        // 2. Join each worker.
        for handle in self.workers.drain(..) {
            // We're already in Drop; can't propagate panics anywhere
            // useful. Best effort: log via `eprintln!`? The crate's
            // `print_stderr` lint forbids that. Swallow silently —
            // worker panics during shutdown are caught by the
            // `done` channel during normal scopes; a panic during
            // a job that races with Drop is a non-issue (the user
            // already saw the scope-level resume_unwind).
            let _ = handle.join();
        }
    }
}

/// Worker thread entry point: pull jobs from the channel until the
/// channel closes, executing each under `catch_unwind`.
fn worker_loop(rx: Receiver<WorkerJob>) {
    while let Ok(job) = rx.recv() {
        // Each job closure carries its own completion signalling
        // logic (captures the per-scope `done_tx` clone). We just
        // run it.
        //
        // Top-level catch_unwind catches panics from the job's own
        // body in case the per-job wrapper (built in
        // `FrameScope::submit`) ever misses one. The per-job
        // wrapper IS supposed to wrap the user closure in
        // catch_unwind and push the payload into the scope's
        // panics Vec, so this outer catch is a belt-and-braces
        // safety net.
        let _ = catch_unwind(AssertUnwindSafe(job));
    }
}

/// Internal: outcome of one job. Currently unit — the per-job
/// closure has already pushed any panic payload into the scope's
/// `panics` Vec and updated the pending counter atomically; the
/// outcome message is only used to wake the scope's main loop on
/// `done_rx.recv()`.
pub(crate) struct WorkerOutcome;

/// Pick the first (lowest-index) worker panic. Returns `Some(idx,
/// payload)` or `None` if no worker panicked.
fn first_worker_panic(
    mut panics: Vec<(usize, Box<dyn std::any::Any + Send + 'static>)>,
) -> Option<(usize, Box<dyn std::any::Any + Send + 'static>)> {
    if panics.is_empty() {
        return None;
    }
    // Sort by submission index ascending; first one wins. Could
    // use `min_by_key` but `sort` is simpler and the vector size
    // is bounded by tile count.
    panics.sort_by_key(|(idx, _)| *idx);
    Some(panics.remove(0))
}

#[cfg(test)]
mod construction_tests {
    use super::*;

    #[test]
    fn worker_pool_spawn_failure_is_returned() {
        let error = WorkerPool::try_new_with_spawner(
            NonZeroUsize::new(2).unwrap(),
            |worker_id, receiver| {
                if worker_id == 0 {
                    Ok(thread::spawn(move || worker_loop(receiver)))
                } else {
                    Err(std::io::Error::other("injected spawn failure"))
                }
            },
        )
        .expect_err("spawn failure must be returned");

        assert_eq!(error.kind(), std::io::ErrorKind::Other);
        assert_eq!(error.to_string(), "injected spawn failure");
    }
}
