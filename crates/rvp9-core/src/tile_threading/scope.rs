//! Per-frame scope handle — submits jobs into a [`WorkerPool`] and
//! blocks at [`WorkerPool::scope`] until they all complete.
//!
//! See [`mod@super`] for the API overview and [`super::WorkerPool`]
//! for the panic-priority protocol.

use crossbeam_channel::Sender;
use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::pool::{WorkerJob, WorkerOutcome, WorkerPool};

/// Per-frame work-submission scope. Construct via
/// [`WorkerPool::scope`]. The `'scope` lifetime ties together every
/// reference the scope holds (pool, per-scope channels, panic list,
/// pending counter) — they all live on `WorkerPool::scope`'s stack
/// frame, so they share one fresh lifetime per call. The `'frame`
/// lifetime is fresh per call (HRTB in `scope()`'s signature) so the
/// caller cannot smuggle the reference past the closure boundary.
///
/// Jobs submitted via [`FrameScope::submit`] are `Send + 'frame` —
/// they may borrow stack data from the `scope()` caller. The
/// `'frame`-erasing `unsafe { mem::transmute }` inside `submit` is
/// sound because `WorkerPool::scope` joins every submitted job before
/// returning, so the borrowed data outlives every worker's access.
pub struct FrameScope<'scope, 'frame> {
    pool: &'scope WorkerPool,
    pending: &'scope AtomicUsize,
    done_tx: &'scope Sender<WorkerOutcome>,
    panics: &'scope Mutex<Vec<(usize, Box<dyn Any + Send + 'static>)>>,
    /// Monotonic per-job index used for "first-panic-by-submission-
    /// order" sorting in the panic-priority protocol.
    next_submission_idx: AtomicUsize,
    /// `&'frame mut ()` PhantomData: ties `'frame` to a mutable
    /// borrow's invariance — caller cannot extend it via subtyping
    /// or sub-cast a `'frame`-bounded reference up to `'static`.
    /// The HRTB in `WorkerPool::scope`'s signature forces `'frame`
    /// to be a fresh lifetime that cannot be smuggled out of the
    /// closure body.
    _frame: PhantomData<&'frame mut ()>,
}

impl<'scope, 'frame> FrameScope<'scope, 'frame> {
    /// Internal constructor. `WorkerPool::scope` is the only call
    /// site; the HRTB on its `f` argument is what makes `'frame`
    /// fresh per call.
    pub(crate) fn new(
        pool: &'scope WorkerPool,
        pending: &'scope AtomicUsize,
        done_tx: &'scope Sender<WorkerOutcome>,
        panics: &'scope Mutex<Vec<(usize, Box<dyn Any + Send + 'static>)>>,
    ) -> Self {
        Self {
            pool,
            pending,
            done_tx,
            panics,
            next_submission_idx: AtomicUsize::new(0),
            _frame: PhantomData,
        }
    }

    /// Submit a job to the pool. The job may borrow data from the
    /// `'frame` enclosing scope.
    ///
    /// Submission is non-blocking; `WorkerPool::scope` blocks on
    /// completion of every submitted job before returning.
    ///
    /// # Panic handling
    ///
    /// If `job` panics, the payload is captured (via
    /// `catch_unwind`) and pushed into the scope's panic list
    /// indexed by submission order. The scope's main loop re-raises
    /// the first-submitted panic when it returns (or drops worker
    /// panics if the scope closure itself also panicked — see
    /// [`WorkerPool::scope`] for the priority protocol).
    pub fn submit<F>(&self, job: F)
    where
        F: FnOnce() + Send + 'frame,
    {
        // Assign a submission index BEFORE incrementing pending so
        // the index → submission-order mapping is stable even if
        // multiple submissions race. `FrameScope` is `Send + Sync`
        // (refs to `Sync` types + `PhantomData<&mut ()>`). Single-
        // submitter is a caller-side discipline, not a type-system
        // guarantee. Atomics ensure multi-submitter remains sound if
        // a future decoder wants it.
        let submission_idx = self.next_submission_idx.fetch_add(1, Ordering::SeqCst);
        self.pending.fetch_add(1, Ordering::SeqCst);

        // Clone the per-scope channels so the closure (which will
        // travel into a worker thread) carries its own handles.
        let done_tx = self.done_tx.clone();
        // SAFETY: `self.panics` is a `&'scope Mutex<...>` to a Mutex
        // owned by `WorkerPool::scope`'s stack frame. Boxing the
        // closure as `Box<dyn FnOnce + 'static>` (the `mem::transmute`
        // further down) requires every capture to be `'static`-ish;
        // we therefore extend `panics_ref` to `'static` here under
        // the SAME contract that bounds the user closure's `'frame`:
        // `WorkerPool::scope` joins every submitted job before
        // returning (the drain loop in `pool.rs::scope` runs
        // unconditionally after `catch_unwind`), so the Mutex
        // outlives every worker access. See SAFETY.md §2 link (e)
        // for the full chain.
        let panics_ref: &'static Mutex<Vec<(usize, Box<dyn Any + Send + 'static>)>> =
            unsafe { std::mem::transmute(self.panics) };

        // Wrap the user closure in a panic-catching, completion-
        // signalling shell.
        //
        // **Liveness invariant** (review M1 fix): `done_tx.send` MUST
        // run unconditionally, even if the wrapper body itself
        // unwinds (e.g., allocator OOM during `Vec::push`). If we
        // miss the send, `pending` never decrements and
        // `WorkerPool::scope`'s drain loop hangs forever — a silent
        // hang, NOT just a deadlock surfaced via `done_rx.recv()`'s
        // disconnect path. We therefore double-wrap: the INNER
        // `catch_unwind` catches user panics; the OUTER
        // `catch_unwind` catches any wrapper panic (allocator OOM,
        // Mutex poison). The `done_tx.send` runs after both — so
        // `pending` always decrements to 0 within finite time. See
        // SAFETY.md §2 link (d).
        let job_with_signal = move || {
            // Outer catch_unwind: protect against panics in the
            // wrapper bookkeeping itself (mutex lock, Vec::push OOM).
            let _ = catch_unwind(AssertUnwindSafe(|| {
                let result: std::thread::Result<()> = catch_unwind(AssertUnwindSafe(job));
                if let Err(payload) = result
                    && let Ok(mut guard) = panics_ref.lock()
                {
                    // Mutex poisoning is essentially impossible
                    // because `lock()`+`push` runs after the inner
                    // catch_unwind; if it does occur, the panic
                    // source is allocator OOM and we have bigger
                    // problems. The outer catch_unwind ensures
                    // `done_tx.send` still fires below.
                    guard.push((submission_idx, payload));
                }
            }));
            // Signal completion UNCONDITIONALLY. If the receiver is
            // gone, the scope already unwound — best-effort send.
            let _ = done_tx.send(WorkerOutcome);
        };

        // Type-erase the closure: `Box<dyn FnOnce + Send + 'frame>`
        // → `Box<dyn FnOnce + Send + 'static>`. The transmute is
        // sound only because `WorkerPool::scope` joins every
        // submitted job before returning. See SAFETY.md.
        let boxed_frame: Box<dyn FnOnce() + Send + 'frame> = Box::new(job_with_signal);
        // SAFETY:
        // (a) The user closure `F: FnOnce() + Send + 'frame` may
        //     borrow data from the scope() caller's stack frame.
        //     The 'frame lifetime expresses that borrow.
        // (b) `WorkerPool::scope` (the only constructor of
        //     `FrameScope`) joins every submitted job before
        //     returning — see pool.rs `scope()`'s drain loop, which
        //     runs unconditionally after the user closure finishes
        //     (normal return AND panic). The drain loop blocks on
        //     `done_rx.recv()` until `pending == 0`; every worker
        //     decrements pending after invoking the user closure
        //     (the `job_with_signal` wrapper above).
        // (c) Therefore, by the time `scope()` returns, no worker
        //     is still accessing the closure's `'frame`-borrowed
        //     data. The 'static cast lets the closure travel
        //     through the `Sender<dyn FnOnce + Send + 'static>`
        //     channel boundary, but the actual borrow lifetime is
        //     bounded by (b)'s join.
        // (d) `mem::forget(scope)` cannot bypass this: the closure-
        //     form scope() API (HRTB `for<'frame> FnOnce(&FrameScope)
        //     -> R`) never hands the caller an owning FrameScope,
        //     only a borrowed reference. The drain loop runs in
        //     `scope()`'s own frame; the caller cannot skip it.
        // (e) Crossing the channel boundary is `Send` because the
        //     boxed dyn carries `Send`; both the user closure and
        //     the `done_tx`/`panics_ptr` captures are Send (Sender
        //     is Send; raw pointer to Mutex is Send by mutex's
        //     Sync; we move owned values into the closure).
        let boxed_static: Box<dyn FnOnce() + Send + 'static> =
            unsafe { std::mem::transmute(boxed_frame) };

        self.pool
            .job_sender()
            .send(boxed_static as WorkerJob)
            .expect("rvp9 WorkerPool: job channel closed mid-scope (worker pool dropped?)");
    }
}

// Tests live alongside the implementation so they can poke at
// pub(crate) details — primarily the `next_submission_idx`
// reset-per-scope invariant.
#[cfg(test)]
mod tests {
    use super::super::pool::WorkerPool;
    use std::num::NonZeroUsize;
    use std::panic::AssertUnwindSafe;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    /// Smoke test: build a pool with N workers, submit one job, see
    /// it run.
    #[test]
    fn single_job_runs() {
        let mut pool = WorkerPool::new(nz(2));
        let counter = AtomicUsize::new(0);
        pool.scope(|scope| {
            scope.submit(|| {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        });
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    /// Submitting N jobs and joining all of them increments a shared
    /// counter exactly N times.
    #[test]
    fn many_jobs_all_join() {
        let mut pool = WorkerPool::new(nz(4));
        let counter = AtomicUsize::new(0);
        const N: usize = 64;
        pool.scope(|scope| {
            for _ in 0..N {
                scope.submit(|| {
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(counter.load(Ordering::SeqCst), N);
    }

    /// Workers may borrow stack-frame data via `'frame`. The
    /// closure-form scope guarantees the borrow outlives the
    /// worker access.
    #[test]
    fn jobs_can_borrow_frame_data() {
        let mut pool = WorkerPool::new(nz(2));
        let shared = Mutex::new(Vec::<u32>::new());
        let shared_ref = &shared;
        pool.scope(|scope| {
            for i in 0..8u32 {
                // `move` so each closure captures its own copy of
                // `i` (Copy) and its own `shared_ref` (Copy
                // reference). Without `move`, the closure would
                // borrow `i` from the loop variable — which is
                // rebound every iteration — and all workers would
                // see the final value.
                scope.submit(move || {
                    shared_ref.lock().unwrap().push(i);
                });
            }
        });
        let mut v = shared.into_inner().unwrap();
        v.sort();
        assert_eq!(v, (0..8).collect::<Vec<_>>());
    }

    /// One submitted job panics; the scope re-raises that panic
    /// after joining all other jobs.
    #[test]
    fn single_job_panic_propagates() {
        let mut pool = WorkerPool::new(nz(2));
        let counter = AtomicUsize::new(0);
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.scope(|scope| {
                scope.submit(|| {
                    counter.fetch_add(1, Ordering::SeqCst);
                });
                scope.submit(|| panic!("worker boom"));
                scope.submit(|| {
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            });
        }));
        assert!(
            result.is_err(),
            "scope should have re-raised the worker panic"
        );
        // The non-panicking jobs must still have run — the drain
        // loop blocks on every submission.
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    /// Multiple jobs panic — the scope re-raises the FIRST-submitted
    /// panic (by submission index), regardless of completion order.
    #[test]
    fn first_panic_by_submission_order_wins() {
        let mut pool = WorkerPool::new(nz(4));
        // Force ordering: a barrier-style coordination so all jobs
        // are spawned before any runs, then the panics can race.
        // We rely on submission-index sorting rather than order
        // determinism — the test asserts the SUBMISSION-ORDER first
        // panic wins, even if a later submission completes first.
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.scope(|scope| {
                scope.submit(|| panic!("FIRST"));
                scope.submit(|| panic!("SECOND"));
                scope.submit(|| panic!("THIRD"));
            });
        }));
        let payload = result.expect_err("expected a panic to propagate");
        let msg = payload
            .downcast_ref::<&'static str>()
            .copied()
            .unwrap_or_default();
        assert_eq!(msg, "FIRST", "first-by-submission-order panic must win");
    }

    /// The closure passed to `scope()` itself panics; worker panics
    /// during the same scope are dropped on the floor — the closure
    /// failure is primary.
    #[test]
    fn closure_panic_dominates_worker_panic() {
        let mut pool = WorkerPool::new(nz(2));
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.scope(|scope| {
                scope.submit(|| panic!("worker"));
                panic!("closure");
            });
        }));
        let payload = result.expect_err("expected a panic to propagate");
        let msg = payload
            .downcast_ref::<&'static str>()
            .copied()
            .unwrap_or_default();
        assert_eq!(msg, "closure", "closure panic must take priority");
    }

    /// Even when the closure panics, all submitted jobs join — so
    /// borrowed data remains valid for the worker's lifetime.
    #[test]
    fn closure_panic_still_joins_workers() {
        let mut pool = WorkerPool::new(nz(2));
        let counter = AtomicUsize::new(0);
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.scope(|scope| {
                for _ in 0..8 {
                    scope.submit(|| {
                        // Synthetic work to make the panic-vs-join
                        // race nontrivial.
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        counter.fetch_add(1, Ordering::SeqCst);
                    });
                }
                panic!("closure boom");
            });
        }));
        assert!(result.is_err(), "closure panic must propagate");
        // Every worker must have run before scope returned.
        assert_eq!(counter.load(Ordering::SeqCst), 8);
    }

    /// `WorkerPool::new(1)` is a valid degenerate construction —
    /// one worker, sequential job execution from the queue.
    #[test]
    fn single_worker_pool_works() {
        let mut pool = WorkerPool::new(nz(1));
        let counter = AtomicUsize::new(0);
        pool.scope(|scope| {
            for _ in 0..10 {
                scope.submit(|| {
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    /// Reusing the pool across multiple scopes is supported and
    /// the pending counter resets per scope.
    #[test]
    fn pool_reusable_across_scopes() {
        let mut pool = WorkerPool::new(nz(2));
        let counter = AtomicUsize::new(0);
        for _ in 0..4 {
            pool.scope(|scope| {
                for _ in 0..3 {
                    scope.submit(|| {
                        counter.fetch_add(1, Ordering::SeqCst);
                    });
                }
            });
        }
        assert_eq!(counter.load(Ordering::SeqCst), 12);
    }

    /// Submitting zero jobs is a valid no-op scope.
    #[test]
    fn empty_scope() {
        let mut pool = WorkerPool::new(nz(2));
        let r = pool.scope(|_scope| 42u32);
        assert_eq!(r, 42);
    }

    /// The scope returns the closure's value when nothing panics.
    #[test]
    fn scope_returns_closure_value() {
        let mut pool = WorkerPool::new(nz(2));
        let sum = AtomicUsize::new(0);
        let sum_ref = &sum;
        let r = pool.scope(|scope| {
            for i in 0..10u32 {
                // `move` captures `i` (Copy) + `sum_ref` (Copy
                // reference). See `jobs_can_borrow_frame_data` for
                // the gotcha rationale.
                scope.submit(move || {
                    sum_ref.fetch_add(i as usize, Ordering::SeqCst);
                });
            }
            "ok"
        });
        assert_eq!(r, "ok");
        assert_eq!(sum.load(Ordering::SeqCst), 45);
    }
}
