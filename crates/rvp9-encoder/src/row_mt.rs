//! Encoder row-multithreading synchronization primitives.
//!
//! VP9 row-MT encodes one superblock row per job. A row may advance only
//! after the row above has completed the same superblock column. [`RowSync`]
//! provides that happens-before edge without exposing shared pixel or mode
//! buffers; those arenas remain the dispatcher's responsibility.

use std::sync::{Condvar, Mutex};

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
    use super::RowSync;
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
}
