//! Encoder row-multithreading synchronization primitives.
//!
//! VP9 row-MT encodes one superblock row per job. A row may advance only
//! after the row above has completed the same superblock column. [`RowSync`]
//! provides that happens-before edge without exposing shared pixel or mode
//! buffers; those arenas remain the dispatcher's responsibility.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use rvp9_core::mode_info::ModeInfo;

/// A row-MT scope was cancelled after one of its jobs failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowSyncCancelled;

impl std::fmt::Display for RowSyncCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("row-MT scope cancelled")
    }
}

impl std::error::Error for RowSyncCancelled {}

/// Immutable causal state published after one superblock finishes.
///
/// The row below imports these bottom reconstruction edges as its top edge.
/// The final three MI rows carry the motion and intra/inter context needed by
/// `MV_REF_BLOCKS`, whose deepest causal lookup is three MI rows above. Left-
/// edge state remains worker-local because each row job processes its
/// superblocks from left to right.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SbBoundary {
    /// Bottom luma row of the completed 64x64 superblock.
    pub bottom_y: [u8; 64],
    /// Bottom U row of the completed 32x32 chroma block.
    pub bottom_u: [u8; 32],
    /// Bottom V row of the completed 32x32 chroma block.
    pub bottom_v: [u8; 32],
    /// Final bottom three rows of eight luma MI cells, row-major.
    pub bottom_mi: [Option<ModeInfo>; 24],
}

/// Failure while publishing or consuming an SB row-MT boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SbBoundaryError {
    /// Another job already published the same `(tile, row, column)` slot.
    AlreadyPublished,
    /// Row progress was visible without its required immutable boundary.
    MissingPublishedBoundary,
    /// A failed job cancelled the frame wavefront.
    Cancelled,
}

impl std::fmt::Display for SbBoundaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyPublished => f.write_str("SB boundary was already published"),
            Self::MissingPublishedBoundary => {
                f.write_str("row progress advanced without an SB boundary")
            }
            Self::Cancelled => f.write_str("SB boundary wait was cancelled"),
        }
    }
}

impl std::error::Error for SbBoundaryError {}

impl From<RowSyncCancelled> for SbBoundaryError {
    fn from(_: RowSyncCancelled) -> Self {
        Self::Cancelled
    }
}

/// Exact-once immutable boundary publication for one encoded frame.
///
/// Slots are indexed by `(tile column, superblock row, tile-local superblock
/// column)`. Tile widths may differ by one superblock, so the flattened
/// storage records a prefix offset for each tile.
#[derive(Debug)]
pub struct SbBoundaryGrid {
    tile_offsets: Box<[usize]>,
    tile_widths: Box<[usize]>,
    sb_rows: usize,
    slots: Box<[OnceLock<SbBoundary>]>,
    published: AtomicUsize,
}

impl SbBoundaryGrid {
    /// Allocate empty boundary slots for one frame.
    ///
    /// # Panics
    ///
    /// Panics if there are no tiles, a tile is empty, or `sb_rows` is zero.
    #[must_use]
    pub fn new(tile_sb_cols: &[usize], sb_rows: usize) -> Self {
        assert!(!tile_sb_cols.is_empty(), "boundary grid requires a tile");
        assert!(sb_rows > 0, "boundary grid requires an SB row");
        assert!(
            tile_sb_cols.iter().all(|&width| width > 0),
            "boundary grid tiles must contain an SB column"
        );
        let mut tile_offsets = Vec::with_capacity(tile_sb_cols.len());
        let mut slot_count = 0usize;
        for &width in tile_sb_cols {
            tile_offsets.push(slot_count);
            slot_count = slot_count
                .checked_add(width.checked_mul(sb_rows).expect("boundary grid overflow"))
                .expect("boundary grid overflow");
        }
        Self {
            tile_offsets: tile_offsets.into_boxed_slice(),
            tile_widths: tile_sb_cols.into(),
            sb_rows,
            slots: (0..slot_count)
                .map(|_| OnceLock::new())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            published: AtomicUsize::new(0),
        }
    }

    /// Publish one completed SB boundary, then advance its row wavefront.
    ///
    /// The immutable `OnceLock` store happens before [`RowSync::write`], so a
    /// successful reader can never observe progress without the boundary.
    ///
    /// # Errors
    ///
    /// Returns [`SbBoundaryError::AlreadyPublished`] for a duplicate slot or
    /// [`SbBoundaryError::Cancelled`] after another job cancels the frame.
    ///
    /// # Panics
    ///
    /// Panics if any coordinate is outside the configured grid.
    pub fn publish_and_advance(
        &self,
        sync: &RowSync,
        tile_col: usize,
        sb_row: usize,
        local_sb_col: usize,
        boundary: SbBoundary,
    ) -> Result<(), SbBoundaryError> {
        let index = self.index(tile_col, sb_row, local_sb_col);
        self.slots[index]
            .set(boundary)
            .map_err(|_| SbBoundaryError::AlreadyPublished)?;
        self.published.fetch_add(1, Ordering::Relaxed);
        sync.write(
            sb_row,
            i32::try_from(local_sb_col).expect("local SB column exceeds i32"),
            i32::try_from(self.tile_widths[tile_col]).expect("tile width exceeds i32"),
        )?;
        Ok(())
    }

    /// Wait for and return the boundary directly above an SB.
    ///
    /// Row zero has no causal predecessor and returns `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns [`SbBoundaryError::Cancelled`] after a job failure. Returns
    /// [`SbBoundaryError::MissingPublishedBoundary`] if row progress advanced
    /// without its required publication, which indicates an internal ordering
    /// defect.
    ///
    /// # Panics
    ///
    /// Panics if any coordinate is outside the configured grid.
    pub fn wait_for_above(
        &self,
        sync: &RowSync,
        tile_col: usize,
        sb_row: usize,
        local_sb_col: usize,
    ) -> Result<Option<&SbBoundary>, SbBoundaryError> {
        let _ = self.index(tile_col, sb_row, local_sb_col);
        sync.read(
            sb_row,
            i32::try_from(local_sb_col).expect("local SB column exceeds i32"),
        )?;
        if sb_row == 0 {
            return Ok(None);
        }
        let above = self.index(tile_col, sb_row - 1, local_sb_col);
        self.slots[above]
            .get()
            .map(Some)
            .ok_or(SbBoundaryError::MissingPublishedBoundary)
    }

    /// Number of exact-once publications completed so far.
    #[must_use]
    pub fn published_boundaries(&self) -> usize {
        self.published.load(Ordering::Relaxed)
    }

    fn index(&self, tile_col: usize, sb_row: usize, local_sb_col: usize) -> usize {
        let width = *self
            .tile_widths
            .get(tile_col)
            .expect("boundary tile column is out of bounds");
        assert!(sb_row < self.sb_rows, "boundary SB row is out of bounds");
        assert!(
            local_sb_col < width,
            "boundary tile-local SB column is out of bounds"
        );
        self.tile_offsets[tile_col] + sb_row * width + local_sb_col
    }
}

/// Mutable view of one row worker's private full-frame reconstruction and MI
/// shadows.
///
/// Workers never share these slices. The view only exports the completed
/// bottom edge of one superblock or imports the immutable edge published by
/// the row above. Pixel coordinates are frame-global; the MI grid may cover a
/// single column tile and therefore carries its global starting column.
pub struct RowWorkerShadow<'a> {
    recon_y: &'a mut [u8],
    recon_u: &'a mut [u8],
    recon_v: &'a mut [u8],
    frame_width: usize,
    frame_height: usize,
    mi_grid: &'a mut [Option<ModeInfo>],
    mi_col_start: usize,
    mi_cols: usize,
}

impl<'a> RowWorkerShadow<'a> {
    /// Bind owned worker buffers to their frame and tile layout.
    ///
    /// # Panics
    ///
    /// Panics if the reconstruction planes are not tight YUV420 buffers for
    /// a positive frame with 64-aligned width and 8-aligned height, or if
    /// the MI grid does not cover a
    /// non-empty, SB-aligned column-tile rectangle in that frame.
    pub fn new(
        recon_y: &'a mut [u8],
        recon_u: &'a mut [u8],
        recon_v: &'a mut [u8],
        frame_width: usize,
        mi_grid: &'a mut [Option<ModeInfo>],
        mi_col_start: usize,
        mi_cols: usize,
    ) -> Self {
        assert!(
            frame_width > 0 && frame_width.is_multiple_of(64),
            "row worker frame width must be a positive multiple of 64"
        );
        assert!(
            recon_y.len().is_multiple_of(frame_width),
            "row worker luma plane must use tight frame stride"
        );
        let frame_height = recon_y.len() / frame_width;
        assert!(
            frame_height > 0 && frame_height.is_multiple_of(8),
            "row worker frame height must be a positive multiple of 8"
        );
        let uv_width = frame_width / 2;
        let uv_height = frame_height / 2;
        assert_eq!(
            recon_u.len(),
            uv_width * uv_height,
            "row worker U plane must be tight 4:2:0"
        );
        assert_eq!(
            recon_v.len(),
            uv_width * uv_height,
            "row worker V plane must be tight 4:2:0"
        );
        assert!(
            mi_cols > 0 && mi_cols.is_multiple_of(8),
            "row worker MI tile width must contain whole superblocks"
        );
        assert!(
            mi_col_start.is_multiple_of(8),
            "row worker MI tile start must be superblock-aligned"
        );
        assert!(
            mi_col_start + mi_cols <= frame_width / 8,
            "row worker MI tile exceeds the frame"
        );
        assert_eq!(
            mi_grid.len(),
            frame_height / 8 * mi_cols,
            "row worker MI grid has the wrong tile shape"
        );
        Self {
            recon_y,
            recon_u,
            recon_v,
            frame_width,
            frame_height,
            mi_grid,
            mi_col_start,
            mi_cols,
        }
    }

    /// Copy one completed superblock's bottom reconstruction and MI edges.
    ///
    /// # Panics
    ///
    /// Panics if the superblock lies outside this worker's frame or tile.
    pub fn export_boundary(&self, sb_row: usize, global_sb_col: usize) -> SbBoundary {
        let (y_row, y_col, uv_row, uv_col, mi_row, local_mi_col) =
            self.boundary_coordinates(sb_row, global_sb_col);
        let mut boundary = SbBoundary {
            bottom_y: [0; 64],
            bottom_u: [0; 32],
            bottom_v: [0; 32],
            bottom_mi: [None; 24],
        };
        let y_start = y_row * self.frame_width + y_col;
        boundary
            .bottom_y
            .copy_from_slice(&self.recon_y[y_start..y_start + 64]);
        let uv_width = self.frame_width / 2;
        let uv_start = uv_row * uv_width + uv_col;
        boundary
            .bottom_u
            .copy_from_slice(&self.recon_u[uv_start..uv_start + 32]);
        boundary
            .bottom_v
            .copy_from_slice(&self.recon_v[uv_start..uv_start + 32]);
        assert!(
            mi_row >= 2,
            "row boundary requires three visible causal MI rows"
        );
        for row in 0..3 {
            let mi_start = (mi_row - 2 + row) * self.mi_cols + local_mi_col;
            boundary.bottom_mi[row * 8..row * 8 + 8]
                .copy_from_slice(&self.mi_grid[mi_start..mi_start + 8]);
        }
        boundary
    }

    /// Import the immutable edge above the requested superblock row.
    ///
    /// # Panics
    ///
    /// Panics for row zero or if the superblock lies outside this worker's
    /// frame or tile.
    pub fn import_above_boundary(
        &mut self,
        sb_row: usize,
        global_sb_col: usize,
        boundary: &SbBoundary,
    ) {
        assert!(sb_row > 0, "the first SB row has no boundary above it");
        let (_, y_col, _, uv_col, _, local_mi_col) =
            self.boundary_coordinates(sb_row, global_sb_col);
        let y_row = sb_row * 64 - 1;
        let y_start = y_row * self.frame_width + y_col;
        self.recon_y[y_start..y_start + 64].copy_from_slice(&boundary.bottom_y);
        let uv_width = self.frame_width / 2;
        let uv_row = sb_row * 32 - 1;
        let uv_start = uv_row * uv_width + uv_col;
        self.recon_u[uv_start..uv_start + 32].copy_from_slice(&boundary.bottom_u);
        self.recon_v[uv_start..uv_start + 32].copy_from_slice(&boundary.bottom_v);
        let mi_row = sb_row * 8 - 3;
        for row in 0..3 {
            let mi_start = (mi_row + row) * self.mi_cols + local_mi_col;
            self.mi_grid[mi_start..mi_start + 8]
                .copy_from_slice(&boundary.bottom_mi[row * 8..row * 8 + 8]);
        }
    }

    fn boundary_coordinates(
        &self,
        sb_row: usize,
        global_sb_col: usize,
    ) -> (usize, usize, usize, usize, usize, usize) {
        assert!(
            sb_row < self.frame_height.div_ceil(64),
            "row worker SB row is out of bounds"
        );
        let global_mi_col = global_sb_col
            .checked_mul(8)
            .expect("row worker SB column overflow");
        let local_mi_col = global_mi_col
            .checked_sub(self.mi_col_start)
            .expect("row worker SB column precedes its tile");
        assert!(
            local_mi_col + 8 <= self.mi_cols,
            "row worker SB column exceeds its tile"
        );
        (
            ((sb_row + 1) * 64).min(self.frame_height) - 1,
            global_sb_col * 64,
            ((sb_row + 1) * 32).min(self.frame_height / 2) - 1,
            global_sb_col * 32,
            ((sb_row + 1) * 8).min(self.frame_height / 8) - 1,
            local_mi_col,
        )
    }
}

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

/// Failure while assembling exact-once row-job ownership after workers join.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowJobOwnershipError {
    /// Two worker results claim the same `(tile, row)` job.
    Duplicate {
        /// Duplicated job.
        job: RowJob,
        /// Worker recorded first.
        existing_worker: usize,
        /// Worker that attempted the duplicate claim.
        duplicate_worker: usize,
    },
    /// One or more scheduled jobs have no worker result.
    Incomplete {
        /// Number of missing jobs.
        missing_jobs: usize,
    },
}

impl std::fmt::Display for RowJobOwnershipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Duplicate {
                job,
                existing_worker,
                duplicate_worker,
            } => write!(
                f,
                "row job ({},{}) belongs to workers {} and {}",
                job.tile_col, job.sb_row, existing_worker, duplicate_worker
            ),
            Self::Incomplete { missing_jobs } => {
                write!(f, "row worker results are missing {missing_jobs} jobs")
            }
        }
    }
}

impl std::error::Error for RowJobOwnershipError {}

/// Exact owner of every `(tile column, superblock row)` worker result.
///
/// Workers append claimed jobs to their private result. After all workers
/// join, the dispatcher assigns those claims here. Duplicate and missing jobs
/// fail before any frame output is merged. Iteration is always tile-column
/// then row order, independent of scheduling or worker completion order.
#[derive(Debug)]
pub struct RowJobOwners {
    tile_sb_starts: Box<[usize]>,
    tile_sb_cols: Box<[usize]>,
    sb_rows: usize,
    owners: Box<[Option<usize>]>,
    assigned: usize,
}

impl RowJobOwners {
    /// Allocate an empty ownership map for one frame.
    ///
    /// # Panics
    ///
    /// Panics if there are no tiles, a tile has no SB column, or `sb_rows`
    /// is zero.
    #[must_use]
    pub fn new(tile_sb_cols: &[usize], sb_rows: usize) -> Self {
        assert!(!tile_sb_cols.is_empty(), "row ownership requires a tile");
        assert!(sb_rows > 0, "row ownership requires an SB row");
        assert!(
            tile_sb_cols.iter().all(|&cols| cols > 0),
            "row ownership tiles must contain an SB column"
        );
        let mut tile_sb_starts = Vec::with_capacity(tile_sb_cols.len());
        let mut sb_start = 0usize;
        for &cols in tile_sb_cols {
            tile_sb_starts.push(sb_start);
            sb_start = sb_start
                .checked_add(cols)
                .expect("row ownership frame width overflow");
        }
        let jobs = tile_sb_cols
            .len()
            .checked_mul(sb_rows)
            .expect("row ownership job count overflow");
        Self {
            tile_sb_starts: tile_sb_starts.into_boxed_slice(),
            tile_sb_cols: tile_sb_cols.into(),
            sb_rows,
            owners: vec![None; jobs].into_boxed_slice(),
            assigned: 0,
        }
    }

    /// Assign one completed job to a worker.
    ///
    /// # Errors
    ///
    /// Returns [`RowJobOwnershipError::Duplicate`] if another result already
    /// owns the job.
    ///
    /// # Panics
    ///
    /// Panics if the job lies outside this frame.
    pub fn assign(&mut self, worker_id: usize, job: RowJob) -> Result<(), RowJobOwnershipError> {
        let index = self.index(job);
        if let Some(existing_worker) = self.owners[index] {
            return Err(RowJobOwnershipError::Duplicate {
                job,
                existing_worker,
                duplicate_worker: worker_id,
            });
        }
        self.owners[index] = Some(worker_id);
        self.assigned += 1;
        Ok(())
    }

    /// Require every scheduled job to have exactly one worker result.
    ///
    /// # Errors
    ///
    /// Returns [`RowJobOwnershipError::Incomplete`] with the number of
    /// unassigned jobs.
    pub fn require_complete(&self) -> Result<(), RowJobOwnershipError> {
        let missing_jobs = self.owners.len() - self.assigned;
        if missing_jobs == 0 {
            Ok(())
        } else {
            Err(RowJobOwnershipError::Incomplete { missing_jobs })
        }
    }

    /// Iterate assigned jobs in deterministic tile-raster order.
    pub fn iter(&self) -> impl Iterator<Item = (RowJob, usize)> + '_ {
        self.owners
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, owner)| {
                owner.map(|worker_id| {
                    let tile_col = index / self.sb_rows;
                    let sb_row = index % self.sb_rows;
                    (RowJob { sb_row, tile_col }, worker_id)
                })
            })
    }

    /// Copy one owned job rectangle from a full-frame worker shadow.
    ///
    /// `cells_per_sb` is 64 for luma pixels, 32 for 4:2:0 chroma, 8 for
    /// MI/MV cells, and 1 for the SB partition/history arrays. Source and
    /// destination use identical tight full-frame layouts.
    ///
    /// # Panics
    ///
    /// Panics if the job is unassigned, the scale is zero, or either slice
    /// does not match the configured frame shape.
    pub fn merge_job_rect<T: Clone>(
        &self,
        destination: &mut [T],
        source: &[T],
        job: RowJob,
        cells_per_sb: usize,
    ) {
        assert_eq!(
            source.len(),
            destination.len(),
            "row merge source and destination shapes differ"
        );
        for range in self.job_rect_ranges(destination.len(), job, cells_per_sb) {
            destination[range.clone()].clone_from_slice(&source[range]);
        }
    }

    /// Move one owned job rectangle from a disposable full-frame worker shadow.
    ///
    /// This is the ownership-preserving counterpart to [`Self::merge_job_rect`].
    /// Each selected source cell is replaced by `T::default()` so callers can
    /// consume a worker result without cloning its heap-backed values.
    ///
    /// # Panics
    ///
    /// Panics under the same shape and ownership conditions as
    /// [`Self::merge_job_rect`].
    pub fn move_job_rect<T: Default>(
        &self,
        destination: &mut [T],
        source: &mut [T],
        job: RowJob,
        cells_per_sb: usize,
    ) {
        assert_eq!(
            source.len(),
            destination.len(),
            "row merge source and destination shapes differ"
        );
        for range in self.job_rect_ranges(destination.len(), job, cells_per_sb) {
            for (destination, source) in destination[range.clone()]
                .iter_mut()
                .zip(&mut source[range])
            {
                *destination = std::mem::take(source);
            }
        }
    }

    fn job_rect_ranges(
        &self,
        destination_len: usize,
        job: RowJob,
        cells_per_sb: usize,
    ) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        assert!(cells_per_sb > 0, "row merge scale must be positive");
        let index = self.index(job);
        assert!(
            self.owners[index].is_some(),
            "cannot merge an unassigned row job"
        );
        let frame_sb_cols: usize = self.tile_sb_cols.iter().sum();
        let row_stride = frame_sb_cols
            .checked_mul(cells_per_sb)
            .expect("row merge stride overflow");
        let padded_rows = self
            .sb_rows
            .checked_mul(cells_per_sb)
            .expect("row merge height overflow");
        let maximum_len = row_stride
            .checked_mul(padded_rows)
            .expect("row merge frame size overflow");
        assert!(
            destination_len <= maximum_len && destination_len.is_multiple_of(row_stride),
            "row merge destination has the wrong frame shape"
        );
        let actual_rows = destination_len / row_stride;
        let col_start = self.tile_sb_starts[job.tile_col] * cells_per_sb;
        let col_len = self.tile_sb_cols[job.tile_col] * cells_per_sb;
        let row_start = job.sb_row * cells_per_sb;
        assert!(
            row_start < actual_rows,
            "row merge job starts outside the visible frame"
        );
        (row_start..(row_start + cells_per_sb).min(actual_rows)).map(move |row| {
            let start = row * row_stride + col_start;
            start..start + col_len
        })
    }

    fn index(&self, job: RowJob) -> usize {
        assert!(
            job.tile_col < self.tile_sb_cols.len(),
            "row ownership tile column is out of bounds"
        );
        assert!(
            job.sb_row < self.sb_rows,
            "row ownership SB row is out of bounds"
        );
        job.tile_col * self.sb_rows + job.sb_row
    }
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

    /// Claim this worker's next job, stealing from the fullest column after
    /// its current column drains.
    ///
    /// `current_tile_col` is updated whenever the worker changes columns.
    /// A single worker therefore drains its initial column in row order, then
    /// drains the remaining columns by largest queue and lowest-column tie
    /// break. Concurrent workers use the same method without duplicating a
    /// claim.
    ///
    /// # Panics
    ///
    /// Panics if `current_tile_col` is outside this context or a queue mutex
    /// was poisoned by a worker panic.
    pub fn claim_for_worker(&self, current_tile_col: &mut usize) -> Option<RowJob> {
        loop {
            if let Some(job) = self.pop_for_column(*current_tile_col) {
                return Some(job);
            }
            *current_tile_col = self.steal_target(*current_tile_col)?;
        }
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

    /// Cancel every tile-column wavefront after any row job fails.
    ///
    /// A worker-scope panic guard must call this before completion is joined,
    /// otherwise a dependant blocked in another column can strand the scope.
    pub fn cancel(&self) {
        for sync in &self.row_syncs {
            sync.cancel();
        }
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
    cancelled: AtomicBool,
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
            cancelled: AtomicBool::new(false),
        }
    }

    /// Create a no-op synchronization object for the tile-MT path.
    #[must_use]
    pub fn dummy() -> Self {
        Self {
            rows: Box::new([]),
            enabled: false,
            cancelled: AtomicBool::new(false),
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
        self.cancelled.store(false, Ordering::Release);
    }

    /// Cancel this tile column and wake every blocked reader.
    ///
    /// Cancellation is idempotent. Mutex poisoning is recovered here because
    /// this method is the failure path that must wake dependants even after a
    /// worker panic.
    pub fn cancel(&self) {
        if !self.enabled {
            return;
        }
        self.cancelled.store(true, Ordering::Release);
        for row in &self.rows {
            let _completed = row
                .completed_col
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            row.changed.notify_all();
        }
    }

    /// Wait until the row above has completed column `col`.
    ///
    /// Row zero has no dependency and returns immediately. Dummy instances
    /// also return immediately.
    ///
    /// # Errors
    ///
    /// Returns [`RowSyncCancelled`] after any job cancels this tile column.
    ///
    /// # Panics
    ///
    /// Panics when `row` is outside the configured range, `col` is negative,
    /// or the preceding row's mutex was poisoned by a worker panic.
    pub fn read(&self, row: usize, col: i32) -> Result<(), RowSyncCancelled> {
        if !self.enabled {
            return Ok(());
        }
        assert!(row < self.rows.len(), "row-MT read row is out of bounds");
        assert!(col >= 0, "row-MT read column must be non-negative");
        if self.cancelled.load(Ordering::Acquire) {
            return Err(RowSyncCancelled);
        }
        if row == 0 {
            return Ok(());
        }

        let preceding = &self.rows[row - 1];
        let mut completed = preceding
            .completed_col
            .lock()
            .expect("RowSync mutex poisoned");
        while *completed < col {
            if self.cancelled.load(Ordering::Acquire) {
                return Err(RowSyncCancelled);
            }
            completed = preceding
                .changed
                .wait(completed)
                .expect("RowSync mutex poisoned");
        }
        if self.cancelled.load(Ordering::Acquire) {
            return Err(RowSyncCancelled);
        }
        Ok(())
    }

    /// Publish completion of `(row, col)` and wake the dependent row.
    ///
    /// Publishing the final column stores `cols + 1`, matching libvpx's
    /// end-of-row sentinel for a sync range of one. This releases any reader
    /// whose last dependency is the completed row.
    ///
    /// # Errors
    ///
    /// Returns [`RowSyncCancelled`] after any job cancels this tile column.
    ///
    /// # Panics
    ///
    /// Panics when the coordinates are invalid, columns are published out of
    /// order, or the row's mutex was poisoned by a worker panic.
    pub fn write(&self, row: usize, col: i32, cols: i32) -> Result<(), RowSyncCancelled> {
        if !self.enabled {
            return Ok(());
        }
        assert!(row < self.rows.len(), "row-MT write row is out of bounds");
        assert!(cols > 0, "row-MT column count must be positive");
        assert!(
            (0..cols).contains(&col),
            "row-MT write column is out of bounds"
        );
        if self.cancelled.load(Ordering::Acquire) {
            return Err(RowSyncCancelled);
        }

        let progress = &self.rows[row];
        let mut completed = progress
            .completed_col
            .lock()
            .expect("RowSync mutex poisoned");
        if self.cancelled.load(Ordering::Acquire) {
            return Err(RowSyncCancelled);
        }
        assert!(
            col > *completed,
            "row-MT columns must be published in increasing order"
        );
        *completed = if col == cols - 1 { cols + 1 } else { col };
        progress.changed.notify_one();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RowJob, RowJobOwners, RowJobOwnershipError, RowMtContext, RowSync, RowWorkerShadow,
        SbBoundary, SbBoundaryError, SbBoundaryGrid,
    };
    use rvp9_core::block::BlockSize;
    use rvp9_core::intra_pred::IntraMode;
    use rvp9_core::mode_info::{BlockMode, ModeInfo};
    use rvp9_core::transforms::TxSize;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn dummy_sync_is_a_no_op_for_arbitrary_coordinates() {
        let sync = RowSync::dummy();
        sync.read(usize::MAX, -1).unwrap();
        sync.write(usize::MAX, -1, -1).unwrap();
        sync.reset();
    }

    fn boundary(seed: u8) -> SbBoundary {
        SbBoundary {
            bottom_y: [seed; 64],
            bottom_u: [seed.wrapping_add(1); 32],
            bottom_v: [seed.wrapping_add(2); 32],
            bottom_mi: [None; 24],
        }
    }

    fn mode_info(seed: u8) -> ModeInfo {
        ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx8x8,
            skip: seed.is_multiple_of(2),
            segment_id: seed % 8,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    #[test]
    fn serial_worker_shadows_handoff_one_superblock_boundary() {
        const WIDTH: usize = 128;
        const HEIGHT: usize = 128;
        const UV_WIDTH: usize = WIDTH / 2;
        const UV_HEIGHT: usize = HEIGHT / 2;
        const MI_COL_START: usize = 8;
        const MI_COLS: usize = 8;
        let mut producer_y = vec![1u8; WIDTH * HEIGHT];
        let mut producer_u = vec![2u8; UV_WIDTH * UV_HEIGHT];
        let mut producer_v = vec![3u8; UV_WIDTH * UV_HEIGHT];
        let mut producer_mi = vec![None; HEIGHT / 8 * MI_COLS];
        for col in 0..64 {
            producer_y[63 * WIDTH + 64 + col] = 10 + col as u8;
        }
        for col in 0..32 {
            producer_u[31 * UV_WIDTH + 32 + col] = 20 + col as u8;
            producer_v[31 * UV_WIDTH + 32 + col] = 30 + col as u8;
        }
        for row in 5..8 {
            for col in 0..8 {
                producer_mi[row * MI_COLS + col] =
                    Some(mode_info(40 + ((row - 5) * 8 + col) as u8));
            }
        }
        let expected = RowWorkerShadow::new(
            &mut producer_y,
            &mut producer_u,
            &mut producer_v,
            WIDTH,
            &mut producer_mi,
            MI_COL_START,
            MI_COLS,
        )
        .export_boundary(0, 1);

        let grid = SbBoundaryGrid::new(&[1, 1], 2);
        let sync = RowSync::new(2);
        grid.publish_and_advance(&sync, 1, 0, 0, expected.clone())
            .unwrap();

        let mut consumer_y = vec![101u8; WIDTH * HEIGHT];
        let mut consumer_u = vec![102u8; UV_WIDTH * UV_HEIGHT];
        let mut consumer_v = vec![103u8; UV_WIDTH * UV_HEIGHT];
        let mut consumer_mi = vec![None; HEIGHT / 8 * MI_COLS];
        let observed = grid.wait_for_above(&sync, 1, 1, 0).unwrap().unwrap();
        RowWorkerShadow::new(
            &mut consumer_y,
            &mut consumer_u,
            &mut consumer_v,
            WIDTH,
            &mut consumer_mi,
            MI_COL_START,
            MI_COLS,
        )
        .import_above_boundary(1, 1, observed);

        assert_eq!(&consumer_y[63 * WIDTH + 64..64 * WIDTH], &expected.bottom_y);
        assert_eq!(consumer_y[63 * WIDTH + 63], 101);
        assert_eq!(consumer_y[64 * WIDTH + 64], 101);
        assert_eq!(
            &consumer_u[31 * UV_WIDTH + 32..32 * UV_WIDTH],
            &expected.bottom_u
        );
        assert_eq!(
            &consumer_v[31 * UV_WIDTH + 32..32 * UV_WIDTH],
            &expected.bottom_v
        );
        assert_eq!(&consumer_mi[5 * MI_COLS..8 * MI_COLS], &expected.bottom_mi);
        assert!(consumer_mi[..5 * MI_COLS].iter().all(Option::is_none));
        assert!(consumer_mi[8 * MI_COLS..].iter().all(Option::is_none));
    }

    #[test]
    fn worker_shadow_exports_visible_partial_bottom_boundary() {
        const WIDTH: usize = 64;
        const HEIGHT: usize = 72;
        let mut y = vec![1u8; WIDTH * HEIGHT];
        let mut u = vec![2u8; (WIDTH / 2) * (HEIGHT / 2)];
        let mut v = vec![3u8; (WIDTH / 2) * (HEIGHT / 2)];
        let mut mi = vec![None; (WIDTH / 8) * (HEIGHT / 8)];
        y[(HEIGHT - 1) * WIDTH..].fill(11);
        u[(HEIGHT / 2 - 1) * (WIDTH / 2)..].fill(12);
        v[(HEIGHT / 2 - 1) * (WIDTH / 2)..].fill(13);
        mi[(HEIGHT / 8 - 3) * (WIDTH / 8)..].fill(Some(mode_info(14)));

        let boundary = RowWorkerShadow::new(&mut y, &mut u, &mut v, WIDTH, &mut mi, 0, 8)
            .export_boundary(1, 0);
        assert_eq!(boundary.bottom_y, [11; 64]);
        assert_eq!(boundary.bottom_u, [12; 32]);
        assert_eq!(boundary.bottom_v, [13; 32]);
        assert_eq!(boundary.bottom_mi, [Some(mode_info(14)); 24]);
    }

    #[test]
    fn boundary_grid_keeps_unequal_tiles_and_rows_disjoint() {
        let grid = SbBoundaryGrid::new(&[2, 3], 2);
        let tile0_sync = RowSync::new(2);
        let tile1_sync = RowSync::new(2);
        let tile0 = boundary(10);
        let tile1 = boundary(20);

        grid.publish_and_advance(&tile0_sync, 0, 0, 1, tile0.clone())
            .unwrap();
        grid.publish_and_advance(&tile1_sync, 1, 0, 0, tile1.clone())
            .unwrap();

        assert_eq!(
            grid.wait_for_above(&tile0_sync, 0, 1, 1).unwrap(),
            Some(&tile0)
        );
        assert_eq!(
            grid.wait_for_above(&tile1_sync, 1, 1, 0).unwrap(),
            Some(&tile1)
        );
        assert_eq!(grid.published_boundaries(), 2);
    }

    #[test]
    fn boundary_slot_is_published_exactly_once_under_race() {
        let grid = Arc::new(SbBoundaryGrid::new(&[1], 1));
        let sync = Arc::new(RowSync::new(1));
        let (result_tx, result_rx) = mpsc::channel();
        let mut publishers = Vec::new();
        for seed in [30, 40] {
            let publisher_grid = Arc::clone(&grid);
            let publisher_sync = Arc::clone(&sync);
            let publisher_tx = result_tx.clone();
            publishers.push(std::thread::spawn(move || {
                publisher_tx
                    .send(publisher_grid.publish_and_advance(
                        &publisher_sync,
                        0,
                        0,
                        0,
                        boundary(seed),
                    ))
                    .unwrap();
            }));
        }
        drop(result_tx);
        for publisher in publishers {
            publisher.join().unwrap();
        }

        let results: Vec<_> = result_rx.into_iter().collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| **result == Err(SbBoundaryError::AlreadyPublished))
                .count(),
            1
        );
        assert_eq!(grid.published_boundaries(), 1);
    }

    #[test]
    fn boundary_publication_happens_before_row_progress() {
        let grid = Arc::new(SbBoundaryGrid::new(&[1], 2));
        let sync = Arc::new(RowSync::new(2));
        let expected = boundary(50);
        let reader_grid = Arc::clone(&grid);
        let reader_sync = Arc::clone(&sync);
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let observed = reader_grid
                .wait_for_above(&reader_sync, 0, 1, 0)
                .unwrap()
                .unwrap();
            done_tx.send(observed.clone()).unwrap();
        });

        started_rx.recv().unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
        grid.publish_and_advance(&sync, 0, 0, 0, expected.clone())
            .unwrap();
        assert_eq!(
            done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            expected
        );
        reader.join().unwrap();
    }

    #[test]
    fn progress_without_boundary_trips_internal_error() {
        let grid = SbBoundaryGrid::new(&[1], 2);
        let sync = RowSync::new(2);
        sync.write(0, 0, 1).unwrap();
        assert_eq!(
            grid.wait_for_above(&sync, 0, 1, 0),
            Err(SbBoundaryError::MissingPublishedBoundary)
        );
    }

    #[test]
    fn cancelled_boundary_wait_returns_without_publication() {
        let grid = SbBoundaryGrid::new(&[1], 2);
        let sync = RowSync::new(2);
        sync.cancel();
        assert_eq!(
            grid.wait_for_above(&sync, 0, 1, 0),
            Err(SbBoundaryError::Cancelled)
        );
        assert_eq!(grid.published_boundaries(), 0);
    }

    #[test]
    fn first_row_never_waits() {
        RowSync::new(2).read(0, 7).unwrap();
    }

    #[test]
    fn dependent_row_waits_for_matching_column() {
        let sync = Arc::new(RowSync::new(2));
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter_sync = Arc::clone(&sync);
        let waiter = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            waiter_sync.read(1, 2).unwrap();
            done_tx.send(()).unwrap();
        });

        started_rx.recv().unwrap();
        sync.write(0, 0, 4).unwrap();
        sync.write(0, 1, 4).unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
        sync.write(0, 2, 4).unwrap();
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn final_column_releases_end_of_row_reader() {
        let sync = Arc::new(RowSync::new(2));
        let (done_tx, done_rx) = mpsc::channel();
        let waiter_sync = Arc::clone(&sync);
        let waiter = std::thread::spawn(move || {
            waiter_sync.read(1, 4).unwrap();
            done_tx.send(()).unwrap();
        });

        for col in 0..4 {
            sync.write(0, col, 4).unwrap();
        }
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn reset_restores_the_dependency() {
        let sync = Arc::new(RowSync::new(2));
        sync.write(0, 0, 1).unwrap();
        sync.read(1, 0).unwrap();
        sync.reset();

        let (done_tx, done_rx) = mpsc::channel();
        let waiter_sync = Arc::clone(&sync);
        let waiter = std::thread::spawn(move || {
            waiter_sync.read(1, 0).unwrap();
            done_tx.send(()).unwrap();
        });
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
        sync.write(0, 0, 1).unwrap();
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn cancellation_wakes_a_blocked_reader() {
        let sync = Arc::new(RowSync::new(2));
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter_sync = Arc::clone(&sync);
        let waiter = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            done_tx.send(waiter_sync.read(1, 0)).unwrap();
        });

        started_rx.recv().unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
        sync.cancel();
        assert_eq!(
            done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            Err(super::RowSyncCancelled)
        );
        waiter.join().unwrap();
    }

    #[test]
    fn reset_clears_cancellation_between_frames() {
        let sync = RowSync::new(1);
        sync.cancel();
        assert_eq!(sync.read(0, 0), Err(super::RowSyncCancelled));
        assert_eq!(sync.write(0, 0, 1), Err(super::RowSyncCancelled));

        sync.reset();
        sync.read(0, 0).unwrap();
        sync.write(0, 0, 1).unwrap();
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

        let panic = std::panic::catch_unwind(|| sync.read(1, 0).unwrap());
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
    fn one_worker_drains_row_jobs_in_deterministic_column_order() {
        let context = RowMtContext::new(3, 2);
        let mut tile_col = context.initial_tile_for_worker(0);
        let mut jobs = Vec::new();
        while let Some(job) = context.claim_for_worker(&mut tile_col) {
            jobs.push(job);
        }
        assert_eq!(
            jobs,
            vec![
                RowJob {
                    sb_row: 0,
                    tile_col: 0
                },
                RowJob {
                    sb_row: 1,
                    tile_col: 0
                },
                RowJob {
                    sb_row: 0,
                    tile_col: 1
                },
                RowJob {
                    sb_row: 1,
                    tile_col: 1
                },
                RowJob {
                    sb_row: 0,
                    tile_col: 2
                },
                RowJob {
                    sb_row: 1,
                    tile_col: 2
                },
            ]
        );
        assert!(context.queues_drained());
        assert_eq!(context.remaining_jobs(), 0);
    }

    #[test]
    fn row_job_owners_merge_two_worker_shadows_without_holes_or_overlap() {
        let mut owners = RowJobOwners::new(&[1, 2], 2);
        let worker_jobs = [
            [
                RowJob {
                    sb_row: 0,
                    tile_col: 0,
                },
                RowJob {
                    sb_row: 1,
                    tile_col: 1,
                },
            ],
            [
                RowJob {
                    sb_row: 1,
                    tile_col: 0,
                },
                RowJob {
                    sb_row: 0,
                    tile_col: 1,
                },
            ],
        ];
        for (worker_id, jobs) in worker_jobs.iter().enumerate() {
            for &job in jobs {
                owners.assign(worker_id, job).unwrap();
            }
        }
        owners.require_complete().unwrap();
        assert_eq!(
            owners.iter().collect::<Vec<_>>(),
            vec![
                (worker_jobs[0][0], 0),
                (worker_jobs[1][0], 1),
                (worker_jobs[1][1], 1),
                (worker_jobs[0][1], 0),
            ]
        );

        for cells_per_sb in [64, 32, 8, 1] {
            let len = 3 * 2 * cells_per_sb * cells_per_sb;
            let sources = [vec![10u8; len], vec![20u8; len]];
            let mut merged = vec![0u8; len];
            for (job, worker_id) in owners.iter() {
                owners.merge_job_rect(&mut merged, &sources[worker_id], job, cells_per_sb);
            }
            let area_per_worker = 3 * cells_per_sb * cells_per_sb;
            assert_eq!(
                merged.iter().filter(|&&value| value == 10).count(),
                area_per_worker
            );
            assert_eq!(
                merged.iter().filter(|&&value| value == 20).count(),
                area_per_worker
            );
            assert!(!merged.contains(&0));
        }
    }

    #[test]
    fn row_job_owners_merge_clamps_partial_bottom_row() {
        let mut owners = RowJobOwners::new(&[1, 1], 2);
        for tile_col in 0..2 {
            owners
                .assign(
                    0,
                    RowJob {
                        sb_row: 0,
                        tile_col,
                    },
                )
                .unwrap();
            owners
                .assign(
                    1,
                    RowJob {
                        sb_row: 1,
                        tile_col,
                    },
                )
                .unwrap();
        }
        owners.require_complete().unwrap();

        for cells_per_sb in [64, 32, 8, 1] {
            let actual_rows = if cells_per_sb == 1 {
                2
            } else {
                cells_per_sb + cells_per_sb / 8
            };
            let row_stride = 2 * cells_per_sb;
            let len = actual_rows * row_stride;
            let sources = [vec![10u8; len], vec![20u8; len]];
            let mut merged = vec![0u8; len];
            for (job, worker_id) in owners.iter() {
                owners.merge_job_rect(&mut merged, &sources[worker_id], job, cells_per_sb);
            }
            assert!(
                merged[..cells_per_sb * row_stride]
                    .iter()
                    .all(|&value| value == 10)
            );
            assert!(
                merged[cells_per_sb * row_stride..]
                    .iter()
                    .all(|&value| value == 20)
            );
        }
    }

    #[test]
    fn row_job_owners_move_preserves_owned_vec_allocations() {
        let mut owners = RowJobOwners::new(&[2, 1], 2);
        for sb_row in 0..2 {
            owners
                .assign(
                    0,
                    RowJob {
                        sb_row,
                        tile_col: 0,
                    },
                )
                .unwrap();
            owners
                .assign(
                    1,
                    RowJob {
                        sb_row,
                        tile_col: 1,
                    },
                )
                .unwrap();
        }
        owners.require_complete().unwrap();

        let mut sources = [
            (10u8..16).map(|value| vec![value]).collect::<Vec<_>>(),
            (20u8..26).map(|value| vec![value]).collect::<Vec<_>>(),
        ];
        let expected_values = [10, 11, 22, 13, 14, 25];
        let expected_pointers = [
            sources[0][0].as_ptr(),
            sources[0][1].as_ptr(),
            sources[1][2].as_ptr(),
            sources[0][3].as_ptr(),
            sources[0][4].as_ptr(),
            sources[1][5].as_ptr(),
        ];
        let mut merged = vec![Vec::<u8>::new(); 6];

        for (job, worker_id) in owners.iter() {
            owners.move_job_rect(&mut merged, &mut sources[worker_id], job, 1);
        }

        assert_eq!(
            merged.iter().map(|value| value[0]).collect::<Vec<_>>(),
            expected_values
        );
        assert_eq!(
            merged.iter().map(Vec::as_ptr).collect::<Vec<_>>(),
            expected_pointers
        );
        assert!(
            [0, 1, 3, 4]
                .into_iter()
                .all(|index| sources[0][index].is_empty())
        );
        assert!([2, 5].into_iter().all(|index| sources[1][index].is_empty()));
        assert!(
            [2, 5]
                .into_iter()
                .all(|index| !sources[0][index].is_empty())
        );
        assert!(
            [0, 1, 3, 4]
                .into_iter()
                .all(|index| !sources[1][index].is_empty())
        );
    }

    #[test]
    fn row_job_owners_reject_duplicates_and_report_missing_jobs() {
        let mut owners = RowJobOwners::new(&[1, 1], 2);
        let job = RowJob {
            sb_row: 1,
            tile_col: 0,
        };
        owners.assign(3, job).unwrap();
        assert_eq!(
            owners.assign(7, job),
            Err(RowJobOwnershipError::Duplicate {
                job,
                existing_worker: 3,
                duplicate_worker: 7,
            })
        );
        assert_eq!(
            owners.require_complete(),
            Err(RowJobOwnershipError::Incomplete { missing_jobs: 3 })
        );
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
        context.row_sync_for_column(0).write(0, 0, 1).unwrap();
        context.row_sync_for_column(0).read(1, 0).unwrap();

        let context = Arc::new(context);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter_context = Arc::clone(&context);
        let waiter = std::thread::spawn(move || {
            waiter_context.row_sync_for_column(1).read(1, 0).unwrap();
            done_tx.send(()).unwrap();
        });
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
        context.row_sync_for_column(1).write(0, 0, 1).unwrap();
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn context_cancellation_reaches_every_tile_column() {
        let context = RowMtContext::new(3, 2);
        context.cancel();
        for tile_col in 0..3 {
            assert_eq!(
                context.row_sync_for_column(tile_col).read(0, 0),
                Err(super::RowSyncCancelled)
            );
        }
    }
}
