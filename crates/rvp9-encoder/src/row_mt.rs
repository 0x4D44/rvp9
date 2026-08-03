//! Encoder row-multithreading synchronization primitives.
//!
//! VP9 row-MT encodes one superblock row per job. A row may advance only
//! after the row above has completed the same superblock column. [`RowSync`]
//! provides that happens-before edge without exposing shared pixel or mode
//! buffers; those arenas remain the dispatcher's responsibility.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

/// One encoder row-MT work item.
///
/// Tile rows are outside the v1 scope, so a job is fully identified by its
/// column tile and superblock row.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RowJob {
    /// Zero-based superblock row in the frame.
    pub sb_row: usize,
    /// Zero-based column-tile index.
    pub tile_col: usize,
}

/// Frame-local row-MT queues and wavefront state.
///
/// Each tile column owns an ordered queue of superblock rows. Workers begin
/// at `worker_id % tile_cols`; after draining that queue they select the
/// column with the largest number of unclaimed rows. Equal loads choose the
/// lowest column index, matching libvpx's deterministic scan order.
#[derive(Debug)]
pub struct RowMtContext {
    queues: Box<[RowJobQueue]>,
    row_syncs: Box<[RowSync]>,
}

#[derive(Debug)]
struct RowJobQueue {
    tile_col: usize,
    inner: Mutex<RowJobQueueInner>,
    drained: AtomicBool,
    remaining_hint: AtomicUsize,
}

#[derive(Debug)]
struct RowJobQueueInner {
    next_row: usize,
    end_row: usize,
}

impl RowMtContext {
    /// Build queues for one frame.
    ///
    /// # Panics
    ///
    /// Panics if `tile_cols` or `sb_rows` is zero.
    #[must_use]
    pub fn new(tile_cols: usize, sb_rows: usize) -> Self {
        assert!(tile_cols > 0, "row-MT requires at least one tile column");
        assert!(sb_rows > 0, "row-MT requires at least one superblock row");
        Self {
            queues: (0..tile_cols)
                .map(|tile_col| RowJobQueue::new(tile_col, sb_rows))
                .collect(),
            row_syncs: (0..tile_cols).map(|_| RowSync::new(sb_rows)).collect(),
        }
    }

    /// Return the initial tile column assigned to `worker_id`.
    #[must_use]
    pub fn initial_tile_for_worker(&self, worker_id: usize) -> usize {
        worker_id % self.queues.len()
    }

    /// Claim the next row from `tile_col`, if one remains.
    ///
    /// # Panics
    ///
    /// Panics if `tile_col` is outside this context or its queue mutex was
    /// poisoned by a worker panic.
    pub fn pop_for_column(&self, tile_col: usize) -> Option<RowJob> {
        self.queues
            .get(tile_col)
            .expect("row-MT tile column is out of bounds")
            .pop()
    }

    /// Select the column with the most unclaimed row jobs.
    ///
    /// Equal loads choose the lowest column index. Returns `None` once every
    /// queue has been claimed.
    ///
    /// # Panics
    ///
    /// Panics if `current` is outside this context.
    #[must_use]
    pub fn steal_target(&self, current: usize) -> Option<usize> {
        assert!(
            current < self.queues.len(),
            "row-MT current tile column is out of bounds"
        );
        let mut best = None;
        for (tile_col, queue) in self.queues.iter().enumerate() {
            let remaining = queue.remaining();
            if remaining == 0 {
                continue;
            }
            match best {
                None => best = Some((tile_col, remaining)),
                Some((best_col, best_remaining))
                    if remaining > best_remaining
                        || (remaining == best_remaining && tile_col < best_col) =>
                {
                    best = Some((tile_col, remaining));
                }
                Some(_) => {}
            }
        }
        best.map(|(tile_col, _)| tile_col)
    }

    /// Access the wavefront synchronization owned by `tile_col`.
    ///
    /// # Panics
    ///
    /// Panics if `tile_col` is outside this context.
    #[must_use]
    pub fn row_sync_for_column(&self, tile_col: usize) -> &RowSync {
        self.row_syncs
            .get(tile_col)
            .expect("row-MT tile column is out of bounds")
    }

    /// Number of unclaimed jobs across every tile column.
    #[must_use]
    pub fn remaining_jobs(&self) -> usize {
        self.queues.iter().map(RowJobQueue::remaining).sum()
    }

    /// Whether every tile-column queue has been claimed.
    #[must_use]
    pub fn queues_drained(&self) -> bool {
        self.queues
            .iter()
            .all(|queue| queue.drained.load(Ordering::Acquire))
    }
}

impl RowJobQueue {
    fn new(tile_col: usize, rows: usize) -> Self {
        Self {
            tile_col,
            inner: Mutex::new(RowJobQueueInner {
                next_row: 0,
                end_row: rows,
            }),
            drained: AtomicBool::new(false),
            remaining_hint: AtomicUsize::new(rows),
        }
    }

    fn pop(&self) -> Option<RowJob> {
        let mut queue = self.inner.lock().expect("row-MT job queue poisoned");
        if queue.next_row == queue.end_row {
            self.drained.store(true, Ordering::Release);
            self.remaining_hint.store(0, Ordering::Release);
            return None;
        }

        let job = RowJob {
            sb_row: queue.next_row,
            tile_col: self.tile_col,
        };
        queue.next_row += 1;
        let remaining = queue.end_row - queue.next_row;
        self.remaining_hint.store(remaining, Ordering::Release);
        if remaining == 0 {
            self.drained.store(true, Ordering::Release);
        }
        Some(job)
    }

    fn remaining(&self) -> usize {
        self.remaining_hint.load(Ordering::Acquire)
    }
}

/// Per-tile-column wavefront synchronization for encoder row-MT.
///
/// This mirrors libvpx's `VP9RowMTSync` with the encoder's fixed sync range
/// of one superblock. Each entry stores the furthest completed column in one
/// superblock row. A value of `-1` means that the row has not started.
#[derive(Debug)]
pub struct RowSync {
    rows: Box<[RowProgress]>,
    enabled: bool,
}

#[derive(Debug)]
struct RowProgress {
    completed_col: Mutex<i32>,
    changed: Condvar,
}

impl RowSync {
    /// Create an enabled wavefront covering `rows` superblock rows.
    ///
    /// # Panics
    ///
    /// Panics when `rows` is zero.
    #[must_use]
    pub fn new(rows: usize) -> Self {
        assert!(rows > 0, "row-MT synchronization requires at least one row");
        Self {
            rows: (0..rows)
                .map(|_| RowProgress {
                    completed_col: Mutex::new(-1),
                    changed: Condvar::new(),
                })
                .collect(),
            enabled: true,
        }
    }

    /// Create a no-op synchronization object for the tile-MT path.
    #[must_use]
    pub fn dummy() -> Self {
        Self {
            rows: Box::new([]),
            enabled: false,
        }
    }

    /// Reset every row to the not-started state before a new frame.
    ///
    /// The caller must invoke this only while no row jobs are active.
    ///
    /// # Panics
    ///
    /// Panics if a row-progress mutex was poisoned by an earlier worker
    /// panic.
    pub fn reset(&self) {
        if !self.enabled {
            return;
        }
        for row in &self.rows {
            *row.completed_col.lock().expect("RowSync mutex poisoned") = -1;
        }
    }

    /// Wait until the row above has completed column `col`.
    ///
    /// Row zero has no dependency and returns immediately. Dummy instances
    /// also return immediately.
    ///
    /// # Panics
    ///
    /// Panics when `row` is outside the configured range, `col` is negative,
    /// or the preceding row's mutex was poisoned by a worker panic.
    pub fn read(&self, row: usize, col: i32) {
        if !self.enabled {
            return;
        }
        assert!(row < self.rows.len(), "row-MT read row is out of bounds");
        assert!(col >= 0, "row-MT read column must be non-negative");
        if row == 0 {
            return;
        }

        let preceding = &self.rows[row - 1];
        let mut completed = preceding
            .completed_col
            .lock()
            .expect("RowSync mutex poisoned");
        while *completed < col {
            completed = preceding
                .changed
                .wait(completed)
                .expect("RowSync mutex poisoned");
        }
    }

    /// Publish completion of `(row, col)` and wake the dependent row.
    ///
    /// Publishing the final column stores `cols + 1`, matching libvpx's
    /// end-of-row sentinel for a sync range of one. This releases any reader
    /// whose last dependency is the completed row.
    ///
    /// # Panics
    ///
    /// Panics when the coordinates are invalid, columns are published out of
    /// order, or the row's mutex was poisoned by a worker panic.
    pub fn write(&self, row: usize, col: i32, cols: i32) {
        if !self.enabled {
            return;
        }
        assert!(row < self.rows.len(), "row-MT write row is out of bounds");
        assert!(cols > 0, "row-MT column count must be positive");
        assert!(
            (0..cols).contains(&col),
            "row-MT write column is out of bounds"
        );

        let progress = &self.rows[row];
        let mut completed = progress
            .completed_col
            .lock()
            .expect("RowSync mutex poisoned");
        assert!(
            col > *completed,
            "row-MT columns must be published in increasing order"
        );
        *completed = if col == cols - 1 { cols + 1 } else { col };
        progress.changed.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::{RowJob, RowMtContext, RowSync};
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn dummy_sync_is_a_no_op_for_arbitrary_coordinates() {
        let sync = RowSync::dummy();
        sync.read(usize::MAX, -1);
        sync.write(usize::MAX, -1, -1);
        sync.reset();
    }

    #[test]
    fn first_row_never_waits() {
        RowSync::new(2).read(0, 7);
    }

    #[test]
    fn dependent_row_waits_for_matching_column() {
        let sync = Arc::new(RowSync::new(2));
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter_sync = Arc::clone(&sync);
        let waiter = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            waiter_sync.read(1, 2);
            done_tx.send(()).unwrap();
        });

        started_rx.recv().unwrap();
        sync.write(0, 0, 4);
        sync.write(0, 1, 4);
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
        sync.write(0, 2, 4);
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn final_column_releases_end_of_row_reader() {
        let sync = Arc::new(RowSync::new(2));
        let (done_tx, done_rx) = mpsc::channel();
        let waiter_sync = Arc::clone(&sync);
        let waiter = std::thread::spawn(move || {
            waiter_sync.read(1, 4);
            done_tx.send(()).unwrap();
        });

        for col in 0..4 {
            sync.write(0, col, 4);
        }
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn reset_restores_the_dependency() {
        let sync = Arc::new(RowSync::new(2));
        sync.write(0, 0, 1);
        sync.read(1, 0);
        sync.reset();

        let (done_tx, done_rx) = mpsc::channel();
        let waiter_sync = Arc::clone(&sync);
        let waiter = std::thread::spawn(move || {
            waiter_sync.read(1, 0);
            done_tx.send(()).unwrap();
        });
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
        sync.write(0, 0, 1);
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn poisoned_progress_panics_instead_of_deadlocking() {
        let sync = Arc::new(RowSync::new(2));
        let poison_sync = Arc::clone(&sync);
        let _ = std::thread::spawn(move || {
            let _guard = poison_sync.rows[0].completed_col.lock().unwrap();
            panic!("poison row progress");
        })
        .join();

        let panic = std::panic::catch_unwind(|| sync.read(1, 0));
        assert!(panic.is_err());
    }

    #[test]
    fn row_jobs_are_ordered_and_claimed_once() {
        let context = RowMtContext::new(3, 4);
        let mut jobs = Vec::new();
        for tile_col in 0..3 {
            while let Some(job) = context.pop_for_column(tile_col) {
                jobs.push(job);
            }
        }

        let expected: Vec<RowJob> = (0..3)
            .flat_map(|tile_col| (0..4).map(move |sb_row| RowJob { sb_row, tile_col }))
            .collect();
        assert_eq!(jobs, expected);
        assert_eq!(context.remaining_jobs(), 0);
        assert!(context.queues_drained());
    }

    #[test]
    fn workers_start_on_round_robin_tile_columns() {
        let context = RowMtContext::new(3, 2);
        let assignments: Vec<usize> = (0..8)
            .map(|worker| context.initial_tile_for_worker(worker))
            .collect();
        assert_eq!(assignments, [0, 1, 2, 0, 1, 2, 0, 1]);
    }

    #[test]
    fn stealing_prefers_largest_queue_then_lowest_column() {
        let context = RowMtContext::new(3, 4);
        assert_eq!(context.steal_target(2), Some(0));

        assert_eq!(context.pop_for_column(2).unwrap().sb_row, 0);
        assert_eq!(context.steal_target(2), Some(0));

        assert_eq!(context.pop_for_column(0).unwrap().sb_row, 0);
        assert_eq!(context.pop_for_column(0).unwrap().sb_row, 1);
        assert_eq!(context.steal_target(0), Some(1));
    }

    #[test]
    fn concurrent_stealing_claims_every_job_exactly_once() {
        let tile_cols = 4;
        let sb_rows = 11;
        let worker_count = 7;
        let context = Arc::new(RowMtContext::new(tile_cols, sb_rows));
        let (job_tx, job_rx) = mpsc::channel();
        let mut workers = Vec::new();
        for worker_id in 0..worker_count {
            let worker_context = Arc::clone(&context);
            let worker_tx = job_tx.clone();
            workers.push(std::thread::spawn(move || {
                let mut tile_col = worker_context.initial_tile_for_worker(worker_id);
                loop {
                    if let Some(job) = worker_context.pop_for_column(tile_col) {
                        worker_tx.send(job).unwrap();
                    } else if let Some(target) = worker_context.steal_target(tile_col) {
                        tile_col = target;
                    } else {
                        break;
                    }
                }
            }));
        }
        drop(job_tx);
        for worker in workers {
            worker.join().unwrap();
        }

        let jobs: Vec<RowJob> = job_rx.into_iter().collect();
        let unique: HashSet<RowJob> = jobs.iter().copied().collect();
        assert_eq!(jobs.len(), tile_cols * sb_rows);
        assert_eq!(unique.len(), jobs.len());
        assert!((0..tile_cols).all(|tile_col| {
            (0..sb_rows).all(|sb_row| unique.contains(&RowJob { sb_row, tile_col }))
        }));
        assert_eq!(context.remaining_jobs(), 0);
        assert!(context.queues_drained());
    }

    #[test]
    fn each_tile_column_owns_independent_wavefront_state() {
        let context = RowMtContext::new(2, 2);
        context.row_sync_for_column(0).write(0, 0, 1);
        context.row_sync_for_column(0).read(1, 0);

        let context = Arc::new(context);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter_context = Arc::clone(&context);
        let waiter = std::thread::spawn(move || {
            waiter_context.row_sync_for_column(1).read(1, 0);
            done_tx.send(()).unwrap();
        });
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
        context.row_sync_for_column(1).write(0, 0, 1);
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();
    }
}
