//! Per-tile mode-info writer (keyframe partition-tree scaffold).
//!
//! Encoder mirror of the decoder's per-tile partition-tree walker
//! (`rvp9_decoder::tile::decode_tile_intra` + `decode_partition` +
//! `decode_block`). v1 scope was every-SB-PARTITION_NONE; S4.13
//! widens the writer to mirror the decoder's full recursive
//! partition tree (HORZ / VERT / SPLIT down to BLOCK_8x8 leaves)
//! while keeping the per-leaf intra-DC-skip residual contract
//! intact.
//!
//! Each leaf block is still required to be intra (any of the 10
//! Y/UV modes) with `skip == true`; segmentation must be disabled.
//! The caller is responsible for emitting a compressed header with
//! `tx_mode == AllowTx32x32` (or any non-`TxModeSelect` mode) so
//! that the per-block `tx_size` is implicit. Any input
//! [`ModeInfo`] grid that violates the scope is rejected with
//! [`ModesWriterError`].
//!
//! ## Partition-tree inference
//!
//! The decoder splats one `ModeInfo` across every 8x8 cell that the
//! leaf covers (see `build_frame_mi_grid` in `rvp9-decoder`). The
//! writer takes advantage of that: at each level of the recursion,
//! it inspects the cell at the top-left mi-coordinate of the current
//! sub-block, reads its `block_size`, and chooses the partition by
//! comparing against `bsize`'s legal subsizes (NONE / HORZ / VERT /
//! SPLIT). For SPLIT the recursion checks each of the four
//! quadrants in turn.
//!
//! ## Frame-edge scope reduction
//!
//! The writer rejects mi-grids whose `(mi_rows, mi_cols)` aren't
//! multiples of 8 via [`ModesWriterError::NonMultipleOfSb`], so the
//! frame-edge cases (`has_rows` or `has_cols` false; or the
//! `hbs == 0` BLOCK_8x8 sub-partitioning branch on the decoder side)
//! never fire. The writer always emits via the full 4-way
//! `read_partition` tree-walk path. When the writer is widened to
//! arbitrary frame sizes, those branches become future work.
//!
//! The BLOCK_8x8 leaf is accepted as a leaf, but BLOCK_8x8 parents
//! always emit `PARTITION_NONE` (no further sub-8 partitioning into
//! 4x4 / 4x8 / 8x4).
//!
//! ## Wire format produced
//!
//! For each parent block visited by [`write_partition_tree`] (top
//! call at `bsize = Block64x64, n4x4_l2 = 4`):
//!
//! 1. **Partition bit-sequence** against `VP9_KF_PARTITION_PROBS[ctx]`:
//!    - `PARTITION_NONE`  → `0`
//!    - `PARTITION_HORZ`  → `1, 0`
//!    - `PARTITION_VERT`  → `1, 1, 0`
//!    - `PARTITION_SPLIT` → `1, 1, 1`
//!
//!    `ctx` mirrors `PartitionCtx::context(mi_row_in_tile,
//!    mi_col_in_tile, bsl)` where `bsl = n8x8_l2 = n4x4_l2 - 1`.
//! 2. **Per-leaf block** at the chosen `subsize`:
//!    - **Segment_id** — skipped (segmentation disabled).
//!    - **Skip flag** — `1` against `fc.skip_probs[skip_ctx]`.
//!    - **Y intra mode** — chosen mode encoded as the
//!      `VP9_INTRA_MODE_TREE` walk against
//!      `VP9_KF_Y_MODE_PROB[above_mode][left_mode]`.
//!    - **UV intra mode** — chosen mode encoded against
//!      `VP9_KF_UV_MODE_PROB[y_mode]`.
//!    - **tx_size** — NOT emitted on the wire (caller's
//!      `tx_mode != TxModeSelect`).
//!    - **Residual tokens** — none (skip is set).
//!
//! ## Caller contract
//!
//! - The compressed header must commit
//!   `tx_mode = AllowTx32x32` (or any non-`TxModeSelect` mode whose
//!   `tx_mode_to_biggest_tx_size` matches each leaf's expected
//!   `tx_size`). The wire format cannot otherwise communicate the
//!   per-block tx-size for skip-only blocks.
//! - The frame must be a keyframe (or intra-only) and segmentation
//!   must be disabled in the uncompressed header.
//! - Frame dimensions must be multiples of 64 pixels (= 8 mi-units).
//!   Partial superblocks at the frame edge are future work.

use super::error::BitstreamError;
use crate::block::{max_txsize, BlockSize, PartitionType, subsize};
use crate::bitstream::compressed_header::tx_mode_to_biggest_tx_size;
use crate::entropy::{BoolWriter, FrameContext, VP9_KF_PARTITION_PROBS, VP9_KF_UV_MODE_PROB,
    VP9_KF_Y_MODE_PROB};
use crate::inter::mode_info::{
    get_intra_inter_context, write_inter_block_mode_info, InterModeReadParams, InterModeWriteError,
};
use crate::inter::mv_ref::{BlockBounds, NeighbourLookup, Position, PrevFrameMvRef};
use crate::intra_pred::IntraMode;
use crate::mode_info::{get_tx_size_context, BlockMode, ModeInfo};
use crate::transforms::{TxMode, TxSize};

/// `MI_BLOCK_SIZE` — each 64×64 superblock covers an 8×8 mode-info
/// grid. Mirrors `rvp9_decoder::tile::MI_BLOCK_SIZE`.
const MI_BLOCK_SIZE: usize = 8;

/// `MI_MASK` — `MI_BLOCK_SIZE - 1`. Used to compute the SB-relative
/// row index when sampling the per-tile left-partition context array.
const MI_MASK: usize = MI_BLOCK_SIZE - 1;

/// Per-`bsl` block-count offset into the partition prob table. Mirrors
/// libvpx `PARTITION_PLOFFSET = 4` and the decoder constant of the
/// same name.
const PARTITION_PLOFFSET: usize = 4;

/// `partition_context_lookup` from libvpx (`vp9_common_data.c:227-244`).
/// For each block size, `(above, left)` is a 4-bit mask indicating
/// which sub-block sizes have "split" recorded in the partition
/// context. Mirrors `rvp9_decoder::tile::PARTITION_CONTEXT_LOOKUP`.
const PARTITION_CONTEXT_LOOKUP: [(u8, u8); 13] = [
    (15, 15), // 4x4
    (15, 14), // 4x8
    (14, 15), // 8x4
    (14, 14), // 8x8
    (14, 12), // 8x16
    (12, 14), // 16x8
    (12, 12), // 16x16
    (12, 8),  // 16x32
    (8, 12),  // 32x16
    (8, 8),   // 32x32
    (8, 0),   // 32x64
    (0, 8),   // 64x32
    (0, 0),   // 64x64
];

/// Per-tile bounds in 8×8 mode-info units. Mirrors
/// `rvp9_decoder::tile::TileBounds` 1-for-1; defined here so callers
/// in `rvp9-core` (which sits below `rvp9-decoder`) can name the type
/// without an upward dependency.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TileBounds {
    /// First mi row in the tile (inclusive).
    pub mi_row_start: usize,
    /// Last mi row in the tile (exclusive).
    pub mi_row_end: usize,
    /// First mi col in the tile (inclusive).
    pub mi_col_start: usize,
    /// Last mi col in the tile (exclusive).
    pub mi_col_end: usize,
}

/// Errors surfaced by the v1 mode-info writer.
///
/// Most variants signal "your input grid wants a feature this v1
/// writer doesn't yet emit." They exist so the encoder's mode driver
/// can detect — at writer-call time — that it has produced a
/// `ModeInfo` shape the wire formatter can't represent, rather than
/// silently emitting a bitstream the decoder will reject.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ModesWriterError {
    /// A cell on the row/column scan was [`None`]. The v1 writer
    /// requires every cell on the tile's superblock grid to be
    /// populated.
    MissingCell {
        /// Row of the missing cell.
        mi_row: usize,
        /// Column of the missing cell.
        mi_col: usize,
    },
    /// The leaf cell's `block_size` doesn't match the partition-tree
    /// derived expected `subsize` at this level of recursion. Either
    /// the encoder's partition decisions were never written into the
    /// mi-grid for this leaf, or the grid is internally inconsistent
    /// (sibling cells inside the same parent block report sizes that
    /// don't recursively decompose to a legal VP9 partition tree).
    UnsupportedBlockSize {
        /// Row of the offending cell.
        mi_row: usize,
        /// Column of the offending cell.
        mi_col: usize,
        /// The actual block size found in the grid.
        block_size: BlockSize,
    },
    /// The mi-grid carries a leaf shape at `(mi_row, mi_col)` that
    /// doesn't match any of the legal subsize choices for the parent
    /// `bsize` (i.e. neither `bsize`, nor `subsize(bsize, HORZ)`,
    /// nor `subsize(bsize, VERT)`, and the SPLIT recursion couldn't
    /// reconcile the four quadrants). This is the
    /// inconsistent-partition-tree rejection — typically the
    /// encoder's mode-decision driver produced a grid that
    /// isn't a valid VP9 partition tree.
    InconsistentPartitionTree {
        /// Row of the offending cell (top-left of the parent).
        mi_row: usize,
        /// Column of the offending cell (top-left of the parent).
        mi_col: usize,
        /// The parent `bsize` whose subsize lookup failed.
        parent_bsize: BlockSize,
        /// The block size found at the top-left cell of the parent.
        leaf_bsize: BlockSize,
    },
    /// A cell carried `mode != Intra { y_mode: Dc, uv_mode: Dc }`.
    /// Either the cell is inter, or it's intra but with a non-DC
    /// mode.
    UnsupportedBlockMode {
        /// Row of the offending cell.
        mi_row: usize,
        /// Column of the offending cell.
        mi_col: usize,
    },
    /// A cell carried `skip == false`. The v1 writer never emits
    /// residual tokens.
    UnsupportedNonSkip {
        /// Row of the offending cell.
        mi_row: usize,
        /// Column of the offending cell.
        mi_col: usize,
    },
    /// A cell carried `segment_id != 0` or `seg_id_predicted == true`.
    /// The v1 writer assumes segmentation is disabled in the
    /// uncompressed header.
    UnsupportedSegmentation {
        /// Row of the offending cell.
        mi_row: usize,
        /// Column of the offending cell.
        mi_col: usize,
    },
    /// The supplied `(mi_rows, mi_cols)` were not a multiple of 8
    /// (one 64×64 superblock per axis). v1 doesn't handle partial
    /// superblocks at the frame edge.
    NonMultipleOfSb {
        /// Frame mi rows.
        mi_rows: u32,
        /// Frame mi cols.
        mi_cols: u32,
    },
    /// Tile bounds were inconsistent with the supplied mi grid.
    InvalidTileBounds,
    /// The [`BoolWriter`] overflowed its destination buffer.
    BufferOverflow,
    /// An inter-frame leaf carried `BlockMode::Intra`. The inter-frame
    /// writer's v1 scope is "pure inter leaves only" — the
    /// intra-in-inter path is deferred until the encoder's intra
    /// mode-info writer is wired.
    UnsupportedIntraInInter {
        /// Row of the offending cell.
        mi_row: usize,
        /// Column of the offending cell.
        mi_col: usize,
    },
    /// An inter-frame leaf reported an error from
    /// [`write_inter_block_mode_info`]. This wraps the
    /// inter-mode writer's own error variants.
    InterModeWrite {
        /// Row of the offending cell.
        mi_row: usize,
        /// Column of the offending cell.
        mi_col: usize,
        /// The underlying inter-mode-writer error.
        source: InterModeWriteError,
    },
}

impl core::fmt::Display for ModesWriterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingCell { mi_row, mi_col } => {
                write!(f, "modes_writer v1: missing cell at ({mi_row}, {mi_col})")
            }
            Self::UnsupportedBlockSize {
                mi_row,
                mi_col,
                block_size,
            } => write!(
                f,
                "modes_writer: leaf cell ({mi_row}, {mi_col}) has block_size {block_size:?} but the partition tree expected a different subsize at this level"
            ),
            Self::InconsistentPartitionTree {
                mi_row,
                mi_col,
                parent_bsize,
                leaf_bsize,
            } => write!(
                f,
                "modes_writer: parent {parent_bsize:?} at ({mi_row}, {mi_col}) cannot host leaf size {leaf_bsize:?} (not a legal HORZ/VERT/NONE subsize)"
            ),
            Self::UnsupportedBlockMode { mi_row, mi_col } => write!(
                f,
                "modes_writer v1: cell ({mi_row}, {mi_col}) requires intra DC_PRED on Y and UV"
            ),
            Self::UnsupportedNonSkip { mi_row, mi_col } => write!(
                f,
                "modes_writer v1: cell ({mi_row}, {mi_col}) has skip=false; v1 emits no residual tokens"
            ),
            Self::UnsupportedSegmentation { mi_row, mi_col } => write!(
                f,
                "modes_writer v1: cell ({mi_row}, {mi_col}) has non-zero segment_id or seg_id_predicted set"
            ),
            Self::NonMultipleOfSb { mi_rows, mi_cols } => write!(
                f,
                "modes_writer v1: frame dims ({mi_rows}, {mi_cols}) mi must be multiples of 8"
            ),
            Self::InvalidTileBounds => f.write_str("modes_writer v1: tile bounds outside frame"),
            Self::BufferOverflow => f.write_str("modes_writer v1: BoolWriter overflowed"),
            Self::UnsupportedIntraInInter { mi_row, mi_col } => write!(
                f,
                "modes_writer v1: inter-frame leaf ({mi_row}, {mi_col}) is intra-in-inter; v1 only emits pure inter leaves"
            ),
            Self::InterModeWrite {
                mi_row,
                mi_col,
                source,
            } => write!(
                f,
                "modes_writer v1: inter-frame leaf ({mi_row}, {mi_col}) failed inter-mode write: {source}"
            ),
        }
    }
}

impl std::error::Error for ModesWriterError {}

impl From<ModesWriterError> for BitstreamError {
    fn from(_: ModesWriterError) -> Self {
        // The bitstream module's umbrella error doesn't yet have a
        // dedicated variant for v1-scope-violation. Most callers want
        // to distinguish "writer scope violation" from "bitstream
        // overflow"; the dedicated `ModesWriterError` type carries
        // the detail, while this `From` impl exists so callers that
        // only care about overflow can swallow the v1 errors through
        // the existing umbrella.
        BitstreamError::CompressedHeaderTooLarge
    }
}

/// Per-tile partition context (encoder side). Mirrors
/// `rvp9_decoder::tile::PartitionCtx` byte-for-byte; the encoder must
/// step the context in exactly the same order the decoder updates it
/// (`above_partition[col]` after every block, `left_partition[row]`
/// per SB row).
#[derive(Clone, Debug)]
struct PartitionCtx {
    above_partition: Vec<u8>,
    left_partition: [u8; MI_BLOCK_SIZE],
}

impl PartitionCtx {
    fn new(tile_mi_cols: usize) -> Self {
        Self {
            above_partition: vec![0u8; tile_mi_cols],
            left_partition: [0u8; MI_BLOCK_SIZE],
        }
    }

    fn reset_left(&mut self) {
        self.left_partition = [0u8; MI_BLOCK_SIZE];
    }

    fn context(&self, mi_row_in_tile: usize, mi_col_in_tile: usize, bsl: usize) -> usize {
        let above = (self.above_partition[mi_col_in_tile] >> bsl) & 1;
        let left = (self.left_partition[mi_row_in_tile & MI_MASK] >> bsl) & 1;
        (usize::from(left) * 2 + usize::from(above)) + bsl * PARTITION_PLOFFSET
    }

    fn update(
        &mut self,
        mi_row_in_tile: usize,
        mi_col_in_tile: usize,
        subsize_bs: BlockSize,
        bw: usize,
    ) {
        let (above_v, left_v) = PARTITION_CONTEXT_LOOKUP[subsize_bs as usize];
        let col_end = (mi_col_in_tile + bw).min(self.above_partition.len());
        for slot in &mut self.above_partition[mi_col_in_tile..col_end] {
            *slot = above_v;
        }
        let mi_row_in_sb = mi_row_in_tile & MI_MASK;
        let row_end = (mi_row_in_sb + bw).min(MI_BLOCK_SIZE);
        for slot in &mut self.left_partition[mi_row_in_sb..row_end] {
            *slot = left_v;
        }
    }
}

/// Lightweight per-block neighbour state used to derive the
/// `(above_mode, left_mode)` keyframe-Y-mode probability slot and the
/// skip context. Conceptually mirrors the subset of
/// `rvp9_decoder::tile_ctx::TileDecodeCtx` the v1 writer needs.
#[derive(Clone, Debug)]
struct NeighbourCtx {
    /// Per-tile-column "most-recent above" descriptor. `None` at the
    /// top of the tile, populated after a block is written.
    above: Vec<Option<NeighbourCell>>,
    /// Per-SB-row "most-recent left" descriptor (one slot per
    /// `mi_row & MI_MASK`). Reset at every new SB row.
    left: [Option<NeighbourCell>; MI_BLOCK_SIZE],
}

#[derive(Copy, Clone, Debug)]
struct NeighbourCell {
    /// Latest `y_mode` for this neighbour, used for the keyframe
    /// `VP9_KF_Y_MODE_PROB[above][left]` slot lookup.
    y_mode: IntraMode,
    /// Latest `skip` bit for this neighbour, used for the skip
    /// context.
    skip: bool,
}

impl NeighbourCtx {
    fn new(tile_mi_cols: usize) -> Self {
        Self {
            above: vec![None; tile_mi_cols],
            left: [None; MI_BLOCK_SIZE],
        }
    }

    fn reset_left(&mut self) {
        self.left = [None; MI_BLOCK_SIZE];
    }

    /// `(above_mode, left_mode, skip_ctx)` snapshot for the block
    /// starting at `(mi_row_in_tile, mi_col_in_tile)`. The keyframe
    /// y-mode lookup treats missing neighbours as `DC_PRED` (libvpx
    /// `vp9_above_block_mode` / `vp9_left_block_mode`); the skip
    /// context counts the number of populated neighbours whose `skip`
    /// bit is `1`.
    fn neighbours_for(
        &self,
        mi_row_in_tile: usize,
        mi_col_in_tile: usize,
    ) -> (IntraMode, IntraMode, usize) {
        let above = self.above.get(mi_col_in_tile).copied().flatten();
        let left = self.left.get(mi_row_in_tile & MI_MASK).copied().flatten();
        let above_mode = above.map_or(IntraMode::Dc, |c| c.y_mode);
        let left_mode = left.map_or(IntraMode::Dc, |c| c.y_mode);
        let skip_ctx =
            usize::from(above.is_some_and(|c| c.skip)) + usize::from(left.is_some_and(|c| c.skip));
        (above_mode, left_mode, skip_ctx)
    }

    /// Splat a block's `(y_mode, skip)` into the above/left arrays
    /// for the span `[mi_col_in_tile .. +x_mis] × [mi_row_in_sb ..
    /// +y_mis]`. Mirrors the decoder's `TileDecodeCtx::write_block`
    /// neighbour-grid update (which writes the full ModeInfo, but
    /// the writer only needs the two fields the y-mode + skip
    /// contexts consume).
    fn write_block(
        &mut self,
        mi_row_in_tile: usize,
        mi_col_in_tile: usize,
        x_mis: usize,
        y_mis: usize,
        y_mode: IntraMode,
        skip: bool,
    ) {
        let cell = NeighbourCell { y_mode, skip };
        let col_end = (mi_col_in_tile + x_mis).min(self.above.len());
        for slot in &mut self.above[mi_col_in_tile..col_end] {
            *slot = Some(cell);
        }
        let mi_row_in_sb = mi_row_in_tile & MI_MASK;
        let row_end = (mi_row_in_sb + y_mis).min(MI_BLOCK_SIZE);
        for slot in &mut self.left[mi_row_in_sb..row_end] {
            *slot = Some(cell);
        }
    }
}

/// Encode a keyframe tile from a fully-decided `Option<ModeInfo>`
/// grid.
///
/// `mi_grid` is laid out row-major over the frame at stride `mi_cols`
/// — the same shape `rvp9_decoder::Decoder::last_frame_mi_grid()`
/// returns. `tile_bounds` selects the rectangle of the grid this
/// tile owns. `fc` provides the per-frame entropy context (the only
/// field the writer reads is `fc.skip_probs`; partitions and modes
/// use the keyframe tables directly).
///
/// Walks `tile_bounds` in row-major 64x64 SB order; for each SB
/// drives [`write_partition_tree`] which recursively emits the
/// partition decisions (mirrored from the decoder's
/// `decode_partition`) and the per-leaf intra-DC-skip block.
///
/// # Caller contract
///
/// - The compressed header must commit `tx_mode = AllowTx32x32` (or
///   any non-`TxModeSelect` mode whose
///   `tx_mode_to_biggest_tx_size` matches each leaf's `tx_size`).
/// - The uncompressed header must commit `segmentation.enabled =
///   false`.
/// - The frame must be a keyframe (or intra-only) and its
///   `(mi_rows, mi_cols)` must be multiples of 8.
///
/// # Errors
///
/// - [`ModesWriterError::MissingCell`] — a cell on the SB scan was
///   [`None`].
/// - [`ModesWriterError::UnsupportedBlockSize`] /
///   [`ModesWriterError::InconsistentPartitionTree`] /
///   `UnsupportedBlockMode` / `UnsupportedNonSkip` /
///   `UnsupportedSegmentation` — the cell's `ModeInfo` exceeded the
///   writer scope.
/// - [`ModesWriterError::NonMultipleOfSb`] / `InvalidTileBounds` —
///   geometry-level rejections.
/// - [`ModesWriterError::BufferOverflow`] — the [`BoolWriter`]
///   overflowed.
pub fn write_tile_intra_keyframe(
    bw: &mut BoolWriter<'_>,
    mi_grid: &[Option<ModeInfo>],
    mi_rows: u32,
    mi_cols: u32,
    tile_bounds: TileBounds,
    fc: &FrameContext,
) -> Result<(), ModesWriterError> {
    if !mi_rows.is_multiple_of(MI_BLOCK_SIZE as u32)
        || !mi_cols.is_multiple_of(MI_BLOCK_SIZE as u32)
    {
        return Err(ModesWriterError::NonMultipleOfSb { mi_rows, mi_cols });
    }
    let mi_rows_us = mi_rows as usize;
    let mi_cols_us = mi_cols as usize;
    if tile_bounds.mi_row_end > mi_rows_us
        || tile_bounds.mi_col_end > mi_cols_us
        || tile_bounds.mi_row_start > tile_bounds.mi_row_end
        || tile_bounds.mi_col_start > tile_bounds.mi_col_end
        || mi_grid.len() != mi_rows_us * mi_cols_us
    {
        return Err(ModesWriterError::InvalidTileBounds);
    }

    let tile_mi_cols = tile_bounds.mi_col_end - tile_bounds.mi_col_start;
    let mut part_ctx = PartitionCtx::new(tile_mi_cols);
    let mut nbr_ctx = NeighbourCtx::new(tile_mi_cols);

    let mut mi_row = tile_bounds.mi_row_start;
    while mi_row < tile_bounds.mi_row_end {
        part_ctx.reset_left();
        nbr_ctx.reset_left();

        let mut mi_col = tile_bounds.mi_col_start;
        while mi_col < tile_bounds.mi_col_end {
            write_partition_tree(
                bw,
                &mut part_ctx,
                &mut nbr_ctx,
                &tile_bounds,
                mi_grid,
                mi_cols_us,
                mi_row,
                mi_col,
                BlockSize::Block64x64,
                4, // n4x4_l2 for 64x64 = log2(16) = 4
                fc,
            )?;
            mi_col += MI_BLOCK_SIZE;
        }
        mi_row += MI_BLOCK_SIZE;
    }

    if bw.has_error() {
        return Err(ModesWriterError::BufferOverflow);
    }
    Ok(())
}

#[inline]
fn mi_grid_get(
    grid: &[Option<ModeInfo>],
    mi_cols: usize,
    mi_row: usize,
    mi_col: usize,
) -> Option<ModeInfo> {
    grid.get(mi_row * mi_cols + mi_col).and_then(|c| *c)
}

/// Inspect the mi-grid at `(mi_row, mi_col)` and decide which
/// [`PartitionType`] the encoder used to subdivide a `bsize` parent
/// block.
///
/// Relies on the decoder's "splat one ModeInfo across every 8x8 cell
/// covered by a leaf" invariant (`build_frame_mi_grid` in
/// `rvp9-decoder`): the top-left cell of any leaf carries its own
/// `block_size`, which is compared against `bsize` and its three
/// legal split subsizes. If the leaf size matches `bsize` itself,
/// the parent was `PARTITION_NONE`; if it matches `subsize(bsize,
/// HORZ)` or `subsize(bsize, VERT)` we report HORZ / VERT
/// respectively; otherwise the parent was `PARTITION_SPLIT` and the
/// caller recurses (which validates the four quadrants).
///
/// Edge cases:
/// - `bsize == Block8x8`: the writer scope reduction pins the
///   parent-8x8 partition to `PARTITION_NONE`. If the leaf at
///   `(mi_row, mi_col)` is anything other than `Block8x8`, return
///   [`ModesWriterError::InconsistentPartitionTree`] — the decoder
///   `hbs == 0` BLOCK_8x8 sub-partitioning into 4x4/4x8/8x4 isn't
///   supported by this scope.
/// - The leaf size doesn't match `bsize`, `subsize(bsize, HORZ)`, or
///   `subsize(bsize, VERT)`, and the parent isn't square enough to
///   SPLIT (i.e. `subsize(bsize, SPLIT)` is `None`): return
///   [`ModesWriterError::InconsistentPartitionTree`].
fn infer_partition(
    mi_grid: &[Option<ModeInfo>],
    mi_cols: usize,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
) -> Result<PartitionType, ModesWriterError> {
    let leaf = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
        .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
    let leaf_bsize = leaf.block_size;

    if leaf_bsize == bsize {
        return Ok(PartitionType::None);
    }

    // BLOCK_8x8 parents: scope-reduced to NONE only. Any non-matching
    // leaf size at parent-8x8 is rejected.
    if bsize == BlockSize::Block8x8 {
        return Err(ModesWriterError::InconsistentPartitionTree {
            mi_row,
            mi_col,
            parent_bsize: bsize,
            leaf_bsize,
        });
    }

    if let Some(horz) = subsize(bsize, PartitionType::Horz)
        && leaf_bsize == horz
    {
        return Ok(PartitionType::Horz);
    }
    if let Some(vert) = subsize(bsize, PartitionType::Vert)
        && leaf_bsize == vert
    {
        return Ok(PartitionType::Vert);
    }

    // The only remaining legal option is SPLIT; verify the parent
    // supports a SPLIT subsize. If not, the grid is malformed.
    if subsize(bsize, PartitionType::Split).is_some() {
        Ok(PartitionType::Split)
    } else {
        Err(ModesWriterError::InconsistentPartitionTree {
            mi_row,
            mi_col,
            parent_bsize: bsize,
            leaf_bsize,
        })
    }
}

/// Emit the partition bit-sequence for `partition` against the
/// supplied 3-entry probability slice. Mirrors the reader-side
/// `mode_info::read_partition` tree walk over `PARTITION_TREE`
/// exactly:
///
/// - `PARTITION_NONE`  → `0`
/// - `PARTITION_HORZ`  → `1, 0`
/// - `PARTITION_VERT`  → `1, 1, 0`
/// - `PARTITION_SPLIT` → `1, 1, 1`
///
/// The keyframe caller passes `&VP9_KF_PARTITION_PROBS[ctx]`; the
/// non-keyframe (inter) caller passes `&fc.partition_prob[ctx]`. Both
/// have the same `[u8; 3]` shape — the reader walks the same tree
/// regardless of which prob bank fed it.
fn write_partition(
    bw: &mut BoolWriter<'_>,
    probs: &[u8; 3],
    partition: PartitionType,
) {
    match partition {
        PartitionType::None => {
            bw.write(0, probs[0]);
        }
        PartitionType::Horz => {
            bw.write(1, probs[0]);
            bw.write(0, probs[1]);
        }
        PartitionType::Vert => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(0, probs[2]);
        }
        PartitionType::Split => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(1, probs[2]);
        }
    }
}

/// Recursively encode the partition tree rooted at `(mi_row, mi_col)`
/// with parent block-size `bsize` and `n4x4_l2 = log2(n_4x4_wide)`.
///
/// Mirrors the decoder's `decode_partition` (`tile.rs:417-559`) but
/// runs in the encoder direction:
/// 1. Compute `n8x8_l2 = n4x4_l2 - 1`, `num_8x8_wh = 1 << n8x8_l2`,
///    `hbs = num_8x8_wh >> 1`, `bsl = n8x8_l2`.
/// 2. Look up `ctx = part_ctx.context(...)` and infer the partition
///    type from the mi-grid via [`infer_partition`].
/// 3. Emit the partition bit-sequence via [`write_partition`].
/// 4. Switch on the partition: NONE/HORZ/VERT call
///    [`write_block_intra_keyframe`] for one (NONE) or two (HORZ/VERT)
///    leaves; SPLIT recurses into the four quadrants.
/// 5. Apply the libvpx partition-context update gate
///    (`bsize >= BLOCK_8X8 && (bsize == BLOCK_8X8 || partition !=
///    SPLIT)`).
///
/// Scope assumptions: `has_rows && has_cols` always hold inside the
/// writer because `write_tile_intra_keyframe` rejects mi-grids whose
/// dimensions aren't multiples of 8. The `hbs == 0` BLOCK_8x8
/// sub-partitioning branch on the decoder side is similarly out of
/// scope; [`infer_partition`] forces BLOCK_8x8 parents to
/// `PARTITION_NONE`.
#[allow(clippy::too_many_arguments)] // mirrors libvpx 1:1
fn write_partition_tree(
    bw: &mut BoolWriter<'_>,
    part_ctx: &mut PartitionCtx,
    nbr_ctx: &mut NeighbourCtx,
    bounds: &TileBounds,
    mi_grid: &[Option<ModeInfo>],
    mi_cols: usize,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    n4x4_l2: u8,
    fc: &FrameContext,
) -> Result<(), ModesWriterError> {
    // Defensive: the frame-edge clip in `decode_partition` should
    // never fire because (1) the writer caller's `NonMultipleOfSb`
    // gate pins `(mi_rows, mi_cols)` to multiples of 8, and (2) the
    // SB scan loop only enters this function at superblock-aligned
    // coordinates inside `tile_bounds`. Express the invariant.
    debug_assert!(mi_row < bounds.mi_row_end && mi_col < bounds.mi_col_end);

    let n8x8_l2: u8 = n4x4_l2.saturating_sub(1);
    let num_8x8_wh: usize = 1usize << n8x8_l2;
    let hbs: usize = num_8x8_wh >> 1;

    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let bsl: usize = n8x8_l2 as usize;
    let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);

    // Frame-edge invariant — the writer is restricted upstream to
    // multiples of 64 px, so the asymmetric (rows-only / cols-only)
    // and (no-rows && no-cols) partition cases never fire. Assert.
    debug_assert!(
        mi_row + hbs < bounds.mi_row_end,
        "has_rows must hold; frame-multiple-of-64 invariant violated"
    );
    debug_assert!(
        mi_col + hbs < bounds.mi_col_end,
        "has_cols must hold; frame-multiple-of-64 invariant violated"
    );

    let partition = infer_partition(mi_grid, mi_cols, mi_row, mi_col, bsize)?;
    write_partition(bw, &VP9_KF_PARTITION_PROBS[ctx], partition);

    let sub = subsize(bsize, partition)
        .expect("infer_partition only returns a partition whose subsize is legal");

    match partition {
        PartitionType::None => {
            // sub == bsize for PARTITION_NONE. Write the leaf.
            let mi = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_block_intra_keyframe(bw, fc, nbr_ctx, bounds, mi_row, mi_col, sub, &mi)?;
        }
        PartitionType::Horz => {
            // Two leaves: (mi_row, mi_col) and (mi_row + hbs, mi_col),
            // both with `sub` = horizontal subsize of `bsize`.
            let top = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_block_intra_keyframe(bw, fc, nbr_ctx, bounds, mi_row, mi_col, sub, &top)?;
            let mi_row_b = mi_row + hbs;
            let bot = mi_grid_get(mi_grid, mi_cols, mi_row_b, mi_col).ok_or(
                ModesWriterError::MissingCell {
                    mi_row: mi_row_b,
                    mi_col,
                },
            )?;
            write_block_intra_keyframe(bw, fc, nbr_ctx, bounds, mi_row_b, mi_col, sub, &bot)?;
        }
        PartitionType::Vert => {
            let left = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_block_intra_keyframe(bw, fc, nbr_ctx, bounds, mi_row, mi_col, sub, &left)?;
            let mi_col_r = mi_col + hbs;
            let right = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col_r).ok_or(
                ModesWriterError::MissingCell {
                    mi_row,
                    mi_col: mi_col_r,
                },
            )?;
            write_block_intra_keyframe(bw, fc, nbr_ctx, bounds, mi_row, mi_col_r, sub, &right)?;
        }
        PartitionType::Split => {
            // 4-way recursion in (mi_row | +hbs, mi_col | +hbs) order
            // — same order as the decoder.
            write_partition_tree(
                bw, part_ctx, nbr_ctx, bounds, mi_grid, mi_cols, mi_row, mi_col, sub, n8x8_l2,
                fc,
            )?;
            write_partition_tree(
                bw,
                part_ctx,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_cols,
                mi_row,
                mi_col + hbs,
                sub,
                n8x8_l2,
                fc,
            )?;
            write_partition_tree(
                bw,
                part_ctx,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_cols,
                mi_row + hbs,
                mi_col,
                sub,
                n8x8_l2,
                fc,
            )?;
            write_partition_tree(
                bw,
                part_ctx,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_cols,
                mi_row + hbs,
                mi_col + hbs,
                sub,
                n8x8_l2,
                fc,
            )?;
        }
    }

    // Partition-context update — mirror libvpx `decode_partition`
    // (`vp9_decodeframe.c:1222-1224`): fires only when `bsize >=
    // BLOCK_8X8 && (bsize == BLOCK_8X8 || partition != SPLIT)`.
    if (bsize as u8) >= (BlockSize::Block8x8 as u8)
        && (bsize == BlockSize::Block8x8 || partition != PartitionType::Split)
    {
        part_ctx.update(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
    }

    Ok(())
}

/// Write the per-leaf keyframe mode info for the block of size
/// `leaf_bsize` starting at `(mi_row, mi_col)`. The leaf is required
/// to be:
///
/// - `block_size == leaf_bsize`,
/// - `mode == Intra { y_mode, uv_mode }` (any of the 10 intra modes),
/// - `skip == true`,
/// - `segment_id == 0 && !seg_id_predicted`.
///
/// Emits, in order:
///
/// 1. `skip = 1` against `fc.skip_probs[skip_ctx]`.
/// 2. `y_mode` against
///    `VP9_KF_Y_MODE_PROB[above_mode][left_mode]` via the
///    `VP9_INTRA_MODE_TREE` walk.
/// 3. `uv_mode` against `VP9_KF_UV_MODE_PROB[y_mode]` (same tree).
///
/// Then updates `nbr_ctx` with the just-written `(y_mode, skip)`
/// pair so subsequent blocks see the correct above/left neighbour
/// state. Mirrors the relevant slice of the decoder's
/// `TileDecodeCtx::write_block`.
#[allow(clippy::too_many_arguments)] // mirrors the decoder's per-block call
fn write_block_intra_keyframe(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    nbr_ctx: &mut NeighbourCtx,
    bounds: &TileBounds,
    mi_row: usize,
    mi_col: usize,
    leaf_bsize: BlockSize,
    mi: &ModeInfo,
) -> Result<(), ModesWriterError> {
    // Scope validation — the leaf cell's recorded block_size must
    // match the partition-tree-derived expected subsize at this
    // level. (`infer_partition` already validated the top-left cell
    // of the parent; here we check the leaf the recursion actually
    // landed on, which for HORZ/VERT/SPLIT is a different cell.)
    if mi.block_size != leaf_bsize {
        return Err(ModesWriterError::UnsupportedBlockSize {
            mi_row,
            mi_col,
            block_size: mi.block_size,
        });
    }
    let (y_mode, uv_mode) = match mi.mode {
        BlockMode::Intra { y_mode, uv_mode } => (y_mode, uv_mode),
        BlockMode::Inter { .. } => {
            return Err(ModesWriterError::UnsupportedBlockMode { mi_row, mi_col });
        }
    };
    if !mi.skip {
        return Err(ModesWriterError::UnsupportedNonSkip { mi_row, mi_col });
    }
    if mi.segment_id != 0 || mi.seg_id_predicted {
        return Err(ModesWriterError::UnsupportedSegmentation { mi_row, mi_col });
    }

    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let (above_mode, left_mode, skip_ctx) =
        nbr_ctx.neighbours_for(mi_row_in_tile, mi_col_in_tile);

    // Segment_id: caller contract pins `segmentation_update_map ==
    // false`, so no bits are emitted.

    // 1. Skip flag — always 1 under writer scope. The skip prob bank
    //    is laid out as `[u8; SKIP_CONTEXTS]` (3 entries) and the
    //    decoder reads with `r.read(probs[ctx])` (`mode_info.rs::
    //    read_skip`).
    bw.write(1, fc.skip_probs[skip_ctx]);

    // 2. Intra Y mode against `VP9_KF_Y_MODE_PROB[above][left]`.
    let y_probs = &VP9_KF_Y_MODE_PROB[above_mode as usize][left_mode as usize];
    write_intra_mode(bw, y_probs, y_mode);

    // 3. Intra UV mode against `VP9_KF_UV_MODE_PROB[y_mode]`.
    let uv_probs = &VP9_KF_UV_MODE_PROB[y_mode as usize];
    write_intra_mode(bw, uv_probs, uv_mode);

    // 4. tx_size — NOT emitted on the wire when `tx_mode !=
    //    TxModeSelect`. The caller commits this via the compressed
    //    header.

    // 5. Splat the just-written neighbour state across the leaf's
    //    mi-coverage. The above/left context for the next block sees
    //    the leaf's Y mode.
    let x_mis = usize::from(leaf_bsize.n_8x8_wide());
    let y_mis = usize::from(leaf_bsize.n_8x8_high());
    nbr_ctx.write_block(mi_row_in_tile, mi_col_in_tile, x_mis, y_mis, y_mode, true);

    Ok(())
}

/// Encode an intra-prediction mode against an
/// `INTRA_MODE_TREE`-shaped 9-prob slot. Mirrors libvpx's
/// `vp9_write_token` against `vp9_intra_mode_tree`
/// (`vp9/common/vp9_entropymode.c:245-255`).
///
/// The bit sequences here are derived statically from
/// `VP9_INTRA_MODE_TREE`. Each `bw.write(bit, prob)` mirrors the
/// reader's `r.read(prob)` and walks the same tree edge. The leaf
/// value `-(mode as i8)` is reached on the last write.
///
/// Used for BOTH:
/// - the keyframe Y mode (`probs = VP9_KF_Y_MODE_PROB[above][left]`)
/// - the keyframe UV mode (`probs = VP9_KF_UV_MODE_PROB[y_mode]`)
/// - (future) non-keyframe Y/UV intra modes against `FrameContext`.
#[inline]
fn write_intra_mode(bw: &mut BoolWriter<'_>, probs: &[u8; 10 - 1], mode: IntraMode) {
    match mode {
        IntraMode::Dc => {
            bw.write(0, probs[0]);
        }
        IntraMode::Tm => {
            bw.write(1, probs[0]);
            bw.write(0, probs[1]);
        }
        IntraMode::V => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(0, probs[2]);
        }
        IntraMode::H => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(1, probs[2]);
            bw.write(0, probs[3]);
            bw.write(0, probs[4]);
        }
        IntraMode::D135 => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(1, probs[2]);
            bw.write(0, probs[3]);
            bw.write(1, probs[4]);
            bw.write(0, probs[5]);
        }
        IntraMode::D117 => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(1, probs[2]);
            bw.write(0, probs[3]);
            bw.write(1, probs[4]);
            bw.write(1, probs[5]);
        }
        IntraMode::D45 => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(1, probs[2]);
            bw.write(1, probs[3]);
            bw.write(0, probs[6]);
        }
        IntraMode::D63 => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(1, probs[2]);
            bw.write(1, probs[3]);
            bw.write(1, probs[6]);
            bw.write(0, probs[7]);
        }
        IntraMode::D153 => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(1, probs[2]);
            bw.write(1, probs[3]);
            bw.write(1, probs[6]);
            bw.write(1, probs[7]);
            bw.write(0, probs[8]);
        }
        IntraMode::D207 => {
            bw.write(1, probs[0]);
            bw.write(1, probs[1]);
            bw.write(1, probs[2]);
            bw.write(1, probs[3]);
            bw.write(1, probs[6]);
            bw.write(1, probs[7]);
            bw.write(1, probs[8]);
        }
    }
}

// ---------------------------------------------------------------------------
// S4.16c — inter-frame partition walker + per-leaf wire emitter.
// ---------------------------------------------------------------------------
//
// Mirror of the decoder's `decode_inter_partition` + `decode_inter_block`.
// v1 scope guards (rejected with `ModesWriterError` variants):
//
// - Frame dims multiples of 64 (= 8 mi-units).
// - Every leaf has `skip == true` (defers residual-coef emission).
// - Every leaf has `segment_id == 0 && !seg_id_predicted` (no segmentation).
// - Every leaf has `BlockMode::Inter { .. }` (no intra-in-inter).
// - Parent inter blocks only (`block_size >= Block8x8`) — sub-8 deferred.

/// `mi_grid`-backed neighbour lookup adapter. Mirrors the decoder's
/// `TileNeighbourLookup` but resolves through a frame-sized
/// `Option<ModeInfo>` grid: the leaf at relative `Position { row, col
/// }` is read from `mi_grid[(base_row + row) * mi_cols + (base_col +
/// col)]` if both indices are non-negative and inside the grid.
///
/// Note the borrow shape: we hold `&'a [Option<ModeInfo>]` and surface
/// `Option<&ModeInfo>` from `get`. Because `NeighbourLookup::get`
/// returns a reference into the grid, the lookup is read-only for the
/// duration of a per-leaf call.
struct MiGridNeighbourLookup<'a> {
    mi_grid: &'a [Option<ModeInfo>],
    mi_cols: usize,
    /// Absolute mi-row of the current block's top-left cell.
    base_mi_row: i32,
    /// Absolute mi-col of the current block's top-left cell.
    base_mi_col: i32,
    /// Frame mi-rows (inclusive bound).
    mi_rows: i32,
    /// Frame mi-cols (inclusive bound).
    mi_cols_frame: i32,
}

impl<'a> MiGridNeighbourLookup<'a> {
    fn new(
        mi_grid: &'a [Option<ModeInfo>],
        mi_cols: usize,
        base_mi_row: usize,
        base_mi_col: usize,
        mi_rows: usize,
    ) -> Self {
        Self {
            mi_grid,
            mi_cols,
            base_mi_row: base_mi_row as i32,
            base_mi_col: base_mi_col as i32,
            mi_rows: mi_rows as i32,
            mi_cols_frame: mi_cols as i32,
        }
    }
}

impl NeighbourLookup for MiGridNeighbourLookup<'_> {
    fn get(&self, pos: Position) -> Option<&ModeInfo> {
        let row = self.base_mi_row.checked_add(i32::from(pos.row))?;
        let col = self.base_mi_col.checked_add(i32::from(pos.col))?;
        if row < 0 || col < 0 || row >= self.mi_rows || col >= self.mi_cols_frame {
            return None;
        }
        let idx = (row as usize) * self.mi_cols + (col as usize);
        self.mi_grid.get(idx)?.as_ref()
    }
}

/// Encode an inter-frame tile from a fully-decided `Option<ModeInfo>`
/// grid. Mirror of [`write_tile_intra_keyframe`] for non-keyframes.
///
/// Walks the tile in 64x64 SB order and drives
/// [`write_inter_partition_tree`] which recursively emits the
/// partition decisions (mirrored from the decoder's
/// `decode_inter_partition`) plus the per-leaf inter block info
/// (segment-id skipped, skip=1, is_inter=1, tx_size, ref frames,
/// inter_mode, switchable_interp, MV diffs).
///
/// Parameters:
/// - `bw`: destination bool-writer.
/// - `fc`: frame context; the writer reads `fc.skip_probs`,
///   `fc.intra_inter_prob`, `fc.partition_prob`, `fc.tx_probs`,
///   plus everything the inter-mode-info writer reads through
///   `fc.{comp_inter,comp_ref,single_ref,inter_mode,switchable_interp,
///   nmvc}_prob*`.
/// - `params`: frame-level inter parameters (allow_hp, interp_filter,
///   reference_mode, sign_bias, comp_fixed/var_ref).
/// - `tx_mode`: compressed-header tx-mode. Non-select modes emit no
///   per-block tx-size bits; `TxModeSelect` emits the tree walk.
/// - `mi_grid`: row-major `mi_rows * mi_cols` grid (decoder splat —
///   every 8x8 cell carries the parent leaf's `ModeInfo`).
/// - `mi_rows` / `mi_cols`: frame dimensions in mi-units.
/// - `tile_bounds`: per-tile rectangle of `mi_grid` this writer
///   processes.
/// - `prev_mv_grid`: optional co-located previous-frame MV grid (one
///   `PrevFrameMvRef` per mi-cell). When `None`, every leaf's
///   `find_mv_refs` call passes `None`, mirroring the
///   "`use_prev_frame_mvs == false`" decoder path.
///
/// # Caller contract
///
/// - The compressed header must commit the same `tx_mode` the caller
///   passes here.
/// - The uncompressed header must commit `segmentation.enabled =
///   false` (or at least `segmentation_update_map = false` so no
///   per-block bits are emitted).
/// - The frame's `(mi_rows, mi_cols)` must be multiples of 8.
/// - The frame must NOT be a keyframe (the keyframe path emits
///   different per-block bits — modes, no is_inter, etc.).
///
/// # Errors
///
/// - [`ModesWriterError::MissingCell`] — a cell on the SB scan was
///   [`None`].
/// - [`ModesWriterError::UnsupportedBlockSize`] /
///   `InconsistentPartitionTree` — partition-tree shape mismatch.
/// - [`ModesWriterError::UnsupportedNonSkip`] /
///   `UnsupportedSegmentation` — per-leaf scope violation.
/// - [`ModesWriterError::UnsupportedIntraInInter`] — leaf is
///   `BlockMode::Intra`.
/// - [`ModesWriterError::InterModeWrite`] — sub-8 leaf, or other
///   inter-mode-writer error.
/// - [`ModesWriterError::NonMultipleOfSb`] /
///   `InvalidTileBounds` — geometry-level rejections.
/// - [`ModesWriterError::BufferOverflow`] — `BoolWriter` overflowed.
#[allow(clippy::too_many_arguments)]
pub fn write_tile_inter(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    params: &InterModeReadParams,
    tx_mode: TxMode,
    mi_grid: &[Option<ModeInfo>],
    mi_rows: u32,
    mi_cols: u32,
    tile_bounds: TileBounds,
    prev_mv_grid: Option<&[PrevFrameMvRef]>,
) -> Result<(), ModesWriterError> {
    if !mi_rows.is_multiple_of(MI_BLOCK_SIZE as u32)
        || !mi_cols.is_multiple_of(MI_BLOCK_SIZE as u32)
    {
        return Err(ModesWriterError::NonMultipleOfSb { mi_rows, mi_cols });
    }
    let mi_rows_us = mi_rows as usize;
    let mi_cols_us = mi_cols as usize;
    if tile_bounds.mi_row_end > mi_rows_us
        || tile_bounds.mi_col_end > mi_cols_us
        || tile_bounds.mi_row_start > tile_bounds.mi_row_end
        || tile_bounds.mi_col_start > tile_bounds.mi_col_end
        || mi_grid.len() != mi_rows_us * mi_cols_us
    {
        return Err(ModesWriterError::InvalidTileBounds);
    }
    if let Some(g) = prev_mv_grid
        && g.len() != mi_rows_us * mi_cols_us
    {
        return Err(ModesWriterError::InvalidTileBounds);
    }

    let tile_mi_cols = tile_bounds.mi_col_end - tile_bounds.mi_col_start;
    let mut part_ctx = PartitionCtx::new(tile_mi_cols);

    let mut mi_row = tile_bounds.mi_row_start;
    while mi_row < tile_bounds.mi_row_end {
        part_ctx.reset_left();

        let mut mi_col = tile_bounds.mi_col_start;
        while mi_col < tile_bounds.mi_col_end {
            write_inter_partition_tree(
                bw,
                fc,
                params,
                tx_mode,
                &mut part_ctx,
                &tile_bounds,
                mi_grid,
                mi_rows_us,
                mi_cols_us,
                mi_row,
                mi_col,
                BlockSize::Block64x64,
                4, // n4x4_l2 for 64x64
                prev_mv_grid,
            )?;
            mi_col += MI_BLOCK_SIZE;
        }
        mi_row += MI_BLOCK_SIZE;
    }

    if bw.has_error() {
        return Err(ModesWriterError::BufferOverflow);
    }
    Ok(())
}

/// Recursively encode the inter-frame partition tree rooted at
/// `(mi_row, mi_col)` with parent block-size `bsize`. Structurally
/// identical to [`write_partition_tree`] (the keyframe variant) but
/// uses `fc.partition_prob[ctx]` instead of
/// `VP9_KF_PARTITION_PROBS[ctx]` and dispatches to
/// [`write_inter_block_leaf`] at the leaves.
#[allow(clippy::too_many_arguments)] // mirrors libvpx 1:1
fn write_inter_partition_tree(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    params: &InterModeReadParams,
    tx_mode: TxMode,
    part_ctx: &mut PartitionCtx,
    bounds: &TileBounds,
    mi_grid: &[Option<ModeInfo>],
    mi_rows: usize,
    mi_cols: usize,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    n4x4_l2: u8,
    prev_mv_grid: Option<&[PrevFrameMvRef]>,
) -> Result<(), ModesWriterError> {
    debug_assert!(mi_row < bounds.mi_row_end && mi_col < bounds.mi_col_end);

    let n8x8_l2: u8 = n4x4_l2.saturating_sub(1);
    let num_8x8_wh: usize = 1usize << n8x8_l2;
    let hbs: usize = num_8x8_wh >> 1;

    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let bsl: usize = n8x8_l2 as usize;
    let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);

    // Same frame-edge invariant as the keyframe writer: dims are
    // multiples of 64, so `has_rows && has_cols` always holds.
    debug_assert!(
        mi_row + hbs < bounds.mi_row_end,
        "has_rows must hold; frame-multiple-of-64 invariant violated"
    );
    debug_assert!(
        mi_col + hbs < bounds.mi_col_end,
        "has_cols must hold; frame-multiple-of-64 invariant violated"
    );

    let partition = infer_partition(mi_grid, mi_cols, mi_row, mi_col, bsize)?;
    write_partition(bw, &fc.partition_prob[ctx], partition);

    let sub = subsize(bsize, partition)
        .expect("infer_partition only returns a partition whose subsize is legal");

    match partition {
        PartitionType::None => {
            let mi = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_inter_block_leaf(
                bw, fc, params, tx_mode, mi_grid, mi_rows, mi_cols, mi_row, mi_col, sub, &mi,
                prev_mv_grid,
            )?;
        }
        PartitionType::Horz => {
            let top = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_inter_block_leaf(
                bw, fc, params, tx_mode, mi_grid, mi_rows, mi_cols, mi_row, mi_col, sub, &top,
                prev_mv_grid,
            )?;
            let mi_row_b = mi_row + hbs;
            let bot = mi_grid_get(mi_grid, mi_cols, mi_row_b, mi_col).ok_or(
                ModesWriterError::MissingCell {
                    mi_row: mi_row_b,
                    mi_col,
                },
            )?;
            write_inter_block_leaf(
                bw, fc, params, tx_mode, mi_grid, mi_rows, mi_cols, mi_row_b, mi_col, sub, &bot,
                prev_mv_grid,
            )?;
        }
        PartitionType::Vert => {
            let left = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_inter_block_leaf(
                bw, fc, params, tx_mode, mi_grid, mi_rows, mi_cols, mi_row, mi_col, sub, &left,
                prev_mv_grid,
            )?;
            let mi_col_r = mi_col + hbs;
            let right = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col_r).ok_or(
                ModesWriterError::MissingCell {
                    mi_row,
                    mi_col: mi_col_r,
                },
            )?;
            write_inter_block_leaf(
                bw, fc, params, tx_mode, mi_grid, mi_rows, mi_cols, mi_row, mi_col_r, sub, &right,
                prev_mv_grid,
            )?;
        }
        PartitionType::Split => {
            write_inter_partition_tree(
                bw, fc, params, tx_mode, part_ctx, bounds, mi_grid, mi_rows, mi_cols, mi_row,
                mi_col, sub, n8x8_l2, prev_mv_grid,
            )?;
            write_inter_partition_tree(
                bw,
                fc,
                params,
                tx_mode,
                part_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row,
                mi_col + hbs,
                sub,
                n8x8_l2,
                prev_mv_grid,
            )?;
            write_inter_partition_tree(
                bw,
                fc,
                params,
                tx_mode,
                part_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row + hbs,
                mi_col,
                sub,
                n8x8_l2,
                prev_mv_grid,
            )?;
            write_inter_partition_tree(
                bw,
                fc,
                params,
                tx_mode,
                part_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row + hbs,
                mi_col + hbs,
                sub,
                n8x8_l2,
                prev_mv_grid,
            )?;
        }
    }

    // Partition-context update gate — same as the keyframe walker.
    if (bsize as u8) >= (BlockSize::Block8x8 as u8)
        && (bsize == BlockSize::Block8x8 || partition != PartitionType::Split)
    {
        part_ctx.update(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
    }

    Ok(())
}

/// Emit one inter-frame leaf coding block at `(mi_row, mi_col)`.
/// Mirror of the decoder's `decode_inter_block` per its doc-comment
/// step list:
///
/// 1. `read_inter_segment_id` — skipped (`segmentation_update_map ==
///    false` is assumed); we validate `segment_id == 0 &&
///    !seg_id_predicted`.
/// 2. `read_skip(ctx)` against `fc.skip_probs[skip_ctx]` — v1 always
///    emits `1`.
/// 3. `read_is_inter_block(ctx)` against `fc.intra_inter_prob[ctx]` —
///    v1 always emits `1`.
/// 4. `read_tx_size` — emits via [`write_tx_size`]. With v1's `skip ==
///    true`, libvpx's `allow_select = !skip || !is_inter` collapses to
///    `false` for inter leaves, so the non-select path fires and no
///    bits are emitted (the decoder caps to `min(max_tx_size,
///    tx_mode_to_biggest_tx_size(tx_mode))`).
/// 5. [`write_inter_block_mode_info`] — ref frames + inter mode +
///    switchable interp + MV diffs.
/// 6. Residual coefs — skipped (v1 sets `skip == true`).
#[allow(clippy::too_many_arguments)]
fn write_inter_block_leaf(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    params: &InterModeReadParams,
    tx_mode: TxMode,
    mi_grid: &[Option<ModeInfo>],
    mi_rows: usize,
    mi_cols: usize,
    mi_row: usize,
    mi_col: usize,
    leaf_bsize: BlockSize,
    mi: &ModeInfo,
    prev_mv_grid: Option<&[PrevFrameMvRef]>,
) -> Result<(), ModesWriterError> {
    // Scope validation: the leaf cell's block_size must match the
    // partition-tree-derived expected subsize at this level.
    if mi.block_size != leaf_bsize {
        return Err(ModesWriterError::UnsupportedBlockSize {
            mi_row,
            mi_col,
            block_size: mi.block_size,
        });
    }
    if mi.segment_id != 0 || mi.seg_id_predicted {
        return Err(ModesWriterError::UnsupportedSegmentation { mi_row, mi_col });
    }
    if !mi.skip {
        return Err(ModesWriterError::UnsupportedNonSkip { mi_row, mi_col });
    }
    if matches!(mi.mode, BlockMode::Intra { .. }) {
        return Err(ModesWriterError::UnsupportedIntraInInter { mi_row, mi_col });
    }

    // Build the neighbour lookup adapter for this leaf. The decoder
    // builds the equivalent `TileNeighbourLookup` rooted at the
    // current block's top-left.
    let neighbours = MiGridNeighbourLookup::new(mi_grid, mi_cols, mi_row, mi_col, mi_rows);
    let above = neighbours.get(Position { row: -1, col: 0 });
    let left = neighbours.get(Position { row: 0, col: -1 });

    // 2. Skip flag — always 1 under writer scope. Mirrors
    //    `decode_inter_block` step 2.
    let skip_ctx = usize::from(above.is_some_and(|m| m.skip))
        + usize::from(left.is_some_and(|m| m.skip));
    bw.write(1, fc.skip_probs[skip_ctx]);

    // 3. is_inter classification — always 1 under writer scope.
    let intra_inter_ctx = get_intra_inter_context(above, left);
    bw.write(1, fc.intra_inter_prob[intra_inter_ctx]);

    // 4. tx_size. `allow_select = !skip || !is_inter` per libvpx
    //    `vp9_decodemv.c:800`. With v1's `skip = true` and `is_inter
    //    = true`, `allow_select = false` so `write_tx_size` emits no
    //    bits.
    let allow_select = !mi.skip;
    let max_tx = max_txsize(leaf_bsize);
    let above_tx = above.map(|m| (m.tx_size, m.skip));
    let left_tx = left.map(|m| (m.tx_size, m.skip));
    let tx_ctx = if max_tx >= TxSize::Tx8x8 {
        get_tx_size_context(above_tx, left_tx, max_tx)
    } else {
        0
    };
    // The decoder, when `allow_select` is false, derives `tx_size =
    // min(max_tx_size, tx_mode_to_biggest_tx_size(tx_mode))`. The
    // `write_tx_size` debug_assert checks this so the caller has to
    // either supply a matching `tx_size` or invoke the select path.
    // Mirror the decoder-side derivation when allow_select == false
    // so the writer accepts any `mi.tx_size` consistent with that
    // derivation. (In `Select` mode we trust the caller's tx_size.)
    let effective_tx_size = if allow_select && tx_mode == TxMode::TxModeSelect {
        mi.tx_size
    } else {
        let cap = tx_mode_to_biggest_tx_size(tx_mode) as u8;
        let raw = core::cmp::min(max_tx as u8, cap);
        match raw {
            0 => TxSize::Tx4x4,
            1 => TxSize::Tx8x8,
            2 => TxSize::Tx16x16,
            3 => TxSize::Tx32x32,
            _ => unreachable!("min of TxSize values out of range: {raw}"),
        }
    };
    crate::mode_info::write_tx_size(
        bw,
        &fc.tx_probs,
        tx_mode,
        max_tx,
        allow_select,
        tx_ctx,
        effective_tx_size,
    );

    // 5. Inter-block mode info. Build the per-leaf `BlockBounds` the
    //    `find_mv_refs` MV clamp inside `write_inter_block_mode_info`
    //    needs (libvpx `set_mi_row_col`).
    let bw_mi = i32::from(leaf_bsize.n_8x8_wide());
    let bh_mi = i32::from(leaf_bsize.n_8x8_high());
    let bounds_block = BlockBounds::new(
        mi_row as i32,
        mi_col as i32,
        mi_rows as i32,
        mi_cols as i32,
        bw_mi,
        bh_mi,
    );
    let prev_mv = prev_mv_grid.and_then(|g| g.get(mi_row * mi_cols + mi_col).copied());
    write_inter_block_mode_info(
        bw,
        fc,
        params,
        &neighbours,
        leaf_bsize,
        &bounds_block,
        prev_mv,
        mi,
    )
    .map_err(|e| ModesWriterError::InterModeWrite {
        mi_row,
        mi_col,
        source: e,
    })?;

    // 6. Residual coefs — skipped (v1 invariant: skip == true).

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::PartitionType;
    use crate::entropy::{BoolReader, FrameContext, VP9_KF_PARTITION_PROBS, VP9_KF_Y_MODE_PROB};
    use crate::mode_info::{read_intra_mode, read_partition, read_skip};
    use crate::transforms::TxSize;

    /// Build a 1-SB (64×64) keyframe mi_grid populated entirely with
    /// the v1-scope-canonical cell.
    fn make_dc_grid(mi_rows: usize, mi_cols: usize) -> Vec<Option<ModeInfo>> {
        let cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        vec![Some(cell); mi_rows * mi_cols]
    }

    /// Mirror of the decoder's per-tile partition-context machinery.
    /// We need it in tests to read back what we just wrote — the
    /// public reader-side helpers in `rvp9_core::mode_info` only do
    /// the per-symbol read; the per-tile context walking is private
    /// to `rvp9_decoder::tile`.
    #[derive(Clone, Debug)]
    struct TestPartCtx {
        above: Vec<u8>,
        left: [u8; MI_BLOCK_SIZE],
    }
    impl TestPartCtx {
        fn new(cols: usize) -> Self {
            Self {
                above: vec![0u8; cols],
                left: [0u8; MI_BLOCK_SIZE],
            }
        }
        fn reset_left(&mut self) {
            self.left = [0u8; MI_BLOCK_SIZE];
        }
        fn context(&self, mi_row_in_tile: usize, mi_col_in_tile: usize, bsl: usize) -> usize {
            let above = (self.above[mi_col_in_tile] >> bsl) & 1;
            let left = (self.left[mi_row_in_tile & MI_MASK] >> bsl) & 1;
            (usize::from(left) * 2 + usize::from(above)) + bsl * PARTITION_PLOFFSET
        }
        fn update(&mut self, mi_row_in_tile: usize, mi_col_in_tile: usize, bw: usize) {
            self.update_with_subsize(mi_row_in_tile, mi_col_in_tile, BlockSize::Block64x64, bw);
        }
        fn update_with_subsize(
            &mut self,
            mi_row_in_tile: usize,
            mi_col_in_tile: usize,
            subsize_bs: BlockSize,
            bw: usize,
        ) {
            let (above_v, left_v) = PARTITION_CONTEXT_LOOKUP[subsize_bs as usize];
            let col_end = (mi_col_in_tile + bw).min(self.above.len());
            for slot in &mut self.above[mi_col_in_tile..col_end] {
                *slot = above_v;
            }
            let mi_row_in_sb = mi_row_in_tile & MI_MASK;
            let row_end = (mi_row_in_sb + bw).min(MI_BLOCK_SIZE);
            for slot in &mut self.left[mi_row_in_sb..row_end] {
                *slot = left_v;
            }
        }
    }

    /// Writer produces the canonical `PARTITION_NONE` 1-bit emission
    /// against the keyframe partition prob table and the round-trip
    /// `read_partition` returns `PARTITION_NONE`.
    #[test]
    fn write_partition_none_round_trips_to_partition_none() {
        let mut buf = vec![0u8; 64];
        let n = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            // Mirror the SB scan for a 64x64 (one-SB) tile.
            let mut part_ctx = PartitionCtx::new(8);
            let bsl = 3usize;
            let ctx = part_ctx.context(0, 0, bsl);
            write_partition(&mut bw, &VP9_KF_PARTITION_PROBS[ctx], PartitionType::None);
            // Mirror the partition-context update the recursive walker
            // does for PARTITION_NONE @ 64x64.
            part_ctx.update(
                0,
                0,
                BlockSize::Block64x64,
                BlockSize::Block64x64.n_8x8_wide() as usize,
            );
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(n);

        let mut reader = BoolReader::new(&buf).expect("reader init");
        let mut test_ctx = TestPartCtx::new(8);
        let ctx = test_ctx.context(0, 0, 3);
        let probs = &VP9_KF_PARTITION_PROBS[ctx];
        let p = read_partition(&mut reader, probs, ctx, None);
        assert_eq!(p, PartitionType::None);
        // Update mirrors the decoder so subsequent SBs see the same
        // context the writer produced.
        test_ctx.update(0, 0, BlockSize::Block64x64.n_8x8_wide() as usize);
        assert_eq!(test_ctx.above[0], 0); // 64x64 → above_v = 0
    }

    /// Writer emits the canonical 3-symbol block (skip=1, y=DC,
    /// uv=DC) and the decoder-side readers recover the same values.
    #[test]
    fn write_block_intra_keyframe_round_trips_dc_dc_skip() {
        let mut buf = vec![0u8; 64];
        let fc = FrameContext::new_default();
        let n = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            let bounds = TileBounds {
                mi_row_start: 0,
                mi_row_end: 8,
                mi_col_start: 0,
                mi_col_end: 8,
            };
            let mut nbr_ctx = NeighbourCtx::new(8);
            let grid = make_dc_grid(8, 8);
            let mi = grid[0].expect("populated cell");
            write_block_intra_keyframe(
                &mut bw,
                &fc,
                &mut nbr_ctx,
                &bounds,
                0,
                0,
                BlockSize::Block64x64,
                &mi,
            )
            .expect("write_block_intra_keyframe should succeed");
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(n);

        let mut reader = BoolReader::new(&buf).expect("reader init");
        // skip_ctx starts at 0 for the top-left block; the reader
        // mirrors the writer's empty above/left context.
        let skip_ctx = 0;
        let skip = read_skip(&mut reader, &fc.skip_probs, skip_ctx, None);
        assert!(skip, "skip should round-trip to true");
        // Y mode reads against VP9_KF_Y_MODE_PROB[Dc][Dc].
        let y_probs = &VP9_KF_Y_MODE_PROB[IntraMode::Dc as usize][IntraMode::Dc as usize];
        let y_mode = read_intra_mode(&mut reader, y_probs);
        assert_eq!(y_mode, IntraMode::Dc);
        // UV mode reads against VP9_KF_UV_MODE_PROB[y_mode].
        let uv_probs = &VP9_KF_UV_MODE_PROB[y_mode as usize];
        let uv_mode = read_intra_mode(&mut reader, uv_probs);
        assert_eq!(uv_mode, IntraMode::Dc);
    }

    /// `write_tile_intra_keyframe` rejects an empty cell.
    #[test]
    fn write_tile_intra_keyframe_rejects_missing_cell() {
        let mut buf = vec![0u8; 64];
        let fc = FrameContext::new_default();
        let mut grid = make_dc_grid(8, 8);
        grid[0] = None;
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_intra_keyframe(&mut bw, &grid, 8, 8, bounds, &fc).unwrap_err();
        assert!(matches!(err, ModesWriterError::MissingCell { mi_row: 0, mi_col: 0 }));
    }

    /// `write_tile_intra_keyframe` rejects a malformed grid where the
    /// top-left cell carries `Block32x32` (implying SPLIT) but the
    /// rest of the SB still carries `Block64x64` cells — the partition
    /// tree can't legally decompose. After the recursion lands in a
    /// 32x32 sub-block whose top-left is still `Block64x64`, the
    /// writer reports `InconsistentPartitionTree`.
    #[test]
    fn write_tile_intra_keyframe_rejects_inconsistent_mixed_sizes() {
        let mut buf = vec![0u8; 64];
        let fc = FrameContext::new_default();
        let mut grid = make_dc_grid(8, 8);
        if let Some(c) = grid[0].as_mut() {
            c.block_size = BlockSize::Block32x32;
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_intra_keyframe(&mut bw, &grid, 8, 8, bounds, &fc).unwrap_err();
        assert!(
            matches!(err, ModesWriterError::InconsistentPartitionTree { .. })
                || matches!(err, ModesWriterError::UnsupportedBlockSize { .. }),
            "got: {err:?}"
        );
    }

    /// `write_tile_intra_keyframe` rejects a non-skip cell.
    #[test]
    fn write_tile_intra_keyframe_rejects_non_skip() {
        let mut buf = vec![0u8; 64];
        let fc = FrameContext::new_default();
        let mut grid = make_dc_grid(8, 8);
        if let Some(c) = grid[0].as_mut() {
            c.skip = false;
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_intra_keyframe(&mut bw, &grid, 8, 8, bounds, &fc).unwrap_err();
        assert!(matches!(err, ModesWriterError::UnsupportedNonSkip { .. }));
    }

    /// `write_tile_intra_keyframe` rejects an inter-mode block. Intra
    /// modes other than DC are now ACCEPTED (S4.12 widened coverage);
    /// the scope rejection applies only to inter mode for the
    /// keyframe-only writer.
    #[test]
    fn write_tile_intra_keyframe_rejects_inter_mode() {
        use rvp9_core_inter_stubs::*;
        let mut buf = vec![0u8; 64];
        let fc = FrameContext::new_default();
        let mut grid = make_dc_grid(8, 8);
        if let Some(c) = grid[0].as_mut() {
            c.mode = synth_inter_mode();
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_intra_keyframe(&mut bw, &grid, 8, 8, bounds, &fc).unwrap_err();
        assert!(matches!(err, ModesWriterError::UnsupportedBlockMode { .. }));
    }

    /// Inline test scaffolding for building an inter-mode `BlockMode`
    /// without depending on the encoder's full inter API. The exact
    /// inter payload values don't matter — we just need a non-Intra
    /// variant for the scope-rejection test above.
    mod rvp9_core_inter_stubs {
        use crate::mode_info::{BlockMode, InterMode, Mv, RefFrame};

        pub fn synth_inter_mode() -> BlockMode {
            BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::IntraFrame, RefFrame::NoRefFrame],
                mvs: [Mv { row: 0, col: 0 }, Mv { row: 0, col: 0 }],
                interp_filter: crate::bitstream::frame::InterpFilter::EightTap,
            }
        }
    }

    /// `write_tile_intra_keyframe` rejects non-multiple-of-64 dims.
    #[test]
    fn write_tile_intra_keyframe_rejects_non_multiple_of_sb() {
        let mut buf = vec![0u8; 64];
        let fc = FrameContext::new_default();
        let grid = make_dc_grid(7, 8); // 56 px high, not a multiple of 64
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 7,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_intra_keyframe(&mut bw, &grid, 7, 8, bounds, &fc).unwrap_err();
        assert!(matches!(err, ModesWriterError::NonMultipleOfSb { .. }));
    }

    /// Full 2x2-SB (128x128) tile round-trips. The partition-context
    /// update path must produce the same `ctx` the reader sees, the
    /// neighbour grid must propagate `(y_mode, skip)` correctly, and
    /// the writer must consume the right number of bits.
    #[test]
    fn write_tile_intra_keyframe_round_trip_2x2_sb_via_per_symbol_readers() {
        // 16x16 mi = 128x128 px = 2x2 superblocks.
        let mi_rows = 16usize;
        let mi_cols = 16usize;
        let grid = make_dc_grid(mi_rows, mi_cols);
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        };
        let fc = FrameContext::new_default();

        let mut buf = vec![0u8; 256];
        let used = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            write_tile_intra_keyframe(
                &mut bw,
                &grid,
                mi_rows as u32,
                mi_cols as u32,
                bounds,
                &fc,
            )
            .expect("write should succeed");
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(used);

        // Replay via the per-symbol readers, walking the same
        // partition + neighbour context the writer used.
        let mut reader = BoolReader::new(&buf).expect("reader init");
        let mut part_ctx = TestPartCtx::new(mi_cols);
        // Neighbour context for the reader-side replay — mirrors the
        // writer's `NeighbourCtx`.
        let mut nbr = NeighbourCtx::new(mi_cols);

        for sb_row in 0..(mi_rows / MI_BLOCK_SIZE) {
            part_ctx.reset_left();
            nbr.reset_left();
            for sb_col in 0..(mi_cols / MI_BLOCK_SIZE) {
                let mi_row = sb_row * MI_BLOCK_SIZE;
                let mi_col = sb_col * MI_BLOCK_SIZE;

                // Partition.
                let ctx = part_ctx.context(mi_row, mi_col, 3);
                let probs = &VP9_KF_PARTITION_PROBS[ctx];
                let p = read_partition(&mut reader, probs, ctx, None);
                assert_eq!(p, PartitionType::None, "sb=({sb_row},{sb_col})");
                part_ctx.update(mi_row, mi_col, MI_BLOCK_SIZE);

                // Skip.
                let (above_mode, left_mode, skip_ctx) = nbr.neighbours_for(mi_row, mi_col);
                let skip = read_skip(&mut reader, &fc.skip_probs, skip_ctx, None);
                assert!(skip, "skip @ ({sb_row},{sb_col})");
                // Y mode.
                let y_probs = &VP9_KF_Y_MODE_PROB[above_mode as usize][left_mode as usize];
                let y_mode = read_intra_mode(&mut reader, y_probs);
                assert_eq!(y_mode, IntraMode::Dc);
                // UV mode.
                let uv_probs = &VP9_KF_UV_MODE_PROB[y_mode as usize];
                let uv_mode = read_intra_mode(&mut reader, uv_probs);
                assert_eq!(uv_mode, IntraMode::Dc);
                let _ = left_mode;

                nbr.write_block(
                    mi_row,
                    mi_col,
                    MI_BLOCK_SIZE,
                    MI_BLOCK_SIZE,
                    IntraMode::Dc,
                    true,
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // S4.12 — widened intra-mode coverage (all 10 modes)
    // -----------------------------------------------------------------

    /// Read an `IntraMode` from a `BoolReader` using the canonical
    /// tree walk against `VP9_INTRA_MODE_TREE`. Mirrors the decoder's
    /// `read_intra_mode` inline at `tile.rs:read_keyframe_y_mode`.
    fn read_intra_mode_via_tree(
        r: &mut crate::entropy::BoolReader<'_>,
        probs: &[u8; 10 - 1],
    ) -> IntraMode {
        let tree: &[i8] = &crate::entropy::adapt::VP9_INTRA_MODE_TREE;
        let leaf = r.read_tree(tree, probs);
        IntraMode::from_u8(leaf as u8).expect("tree leaf must map to a valid IntraMode")
    }

    /// Round-trip every intra mode against a fixed probability set.
    /// Picks `VP9_KF_Y_MODE_PROB[DC][DC]` for stability — any prob set
    /// works since the writer just walks the tree topology.
    #[test]
    fn write_intra_mode_round_trips_for_all_10_modes() {
        for mode in [
            IntraMode::Dc,
            IntraMode::V,
            IntraMode::H,
            IntraMode::D45,
            IntraMode::D135,
            IntraMode::D117,
            IntraMode::D153,
            IntraMode::D207,
            IntraMode::D63,
            IntraMode::Tm,
        ] {
            let probs = &VP9_KF_Y_MODE_PROB[IntraMode::Dc as usize][IntraMode::Dc as usize];
            let mut buf = vec![0u8; 64];
            let written = {
                let mut bw = crate::entropy::BoolWriter::start_encode(&mut buf);
                write_intra_mode(&mut bw, probs, mode);
                bw.stop_encode().expect("stop_encode")
            };
            buf.truncate(written);
            let mut r = crate::entropy::BoolReader::new(&buf).expect("BoolReader::new");
            let got = read_intra_mode_via_tree(&mut r, probs);
            assert_eq!(got, mode, "round-trip failed for {mode:?}");
        }
    }

    /// End-to-end: encode a 1-SB keyframe whose Y/UV modes are each
    /// non-DC; assert the bitstream parses back via the keyframe
    /// Y-mode + UV-mode trees.
    #[test]
    fn write_block_intra_keyframe_accepts_all_intra_modes() {
        use crate::entropy::{BoolReader, BoolWriter, FrameContext};
        let fc = FrameContext::new_default();
        let probs_for_above_left = |above: IntraMode, left: IntraMode| {
            VP9_KF_Y_MODE_PROB[above as usize][left as usize]
        };

        for y_mode in [
            IntraMode::V,
            IntraMode::H,
            IntraMode::Tm,
            IntraMode::D45,
            IntraMode::D135,
            IntraMode::D207,
        ] {
            for uv_mode in [IntraMode::Dc, IntraMode::V, IntraMode::Tm] {
                let mi_rows = 8u32;
                let mi_cols = 8u32;
                let mi = ModeInfo {
                    block_size: BlockSize::Block64x64,
                    tx_size: TxSize::Tx32x32,
                    skip: true,
                    segment_id: 0,
                    seg_id_predicted: false,
                    mode: BlockMode::Intra { y_mode, uv_mode },
                    bmi: Default::default(),
                };
                let grid: Vec<Option<ModeInfo>> = vec![Some(mi); (mi_rows * mi_cols) as usize];
                let bounds = TileBounds {
                    mi_row_start: 0,
                    mi_row_end: 8,
                    mi_col_start: 0,
                    mi_col_end: 8,
                };
                let mut buf = vec![0u8; 2 * 1024];
                let written = {
                    let mut bw = BoolWriter::start_encode(&mut buf);
                    write_tile_intra_keyframe(&mut bw, &grid, mi_rows, mi_cols, bounds, &fc)
                        .expect("write_tile_intra_keyframe");
                    bw.stop_encode().expect("stop_encode")
                };
                buf.truncate(written);

                let mut r = BoolReader::new(&buf).expect("BoolReader::new");
                // 1-bit PARTITION_NONE @ VP9_KF_PARTITION_PROBS[ctx][0].
                // For the first 64x64 SB (no above/left history) the
                // context is `bsl * PARTITION_PLOFFSET = 3*4 = 12`.
                let ctx = 3 * 4;
                let _ = r.read(VP9_KF_PARTITION_PROBS[ctx][0]);
                // skip flag @ skip_probs[0]
                let _ = r.read(fc.skip_probs[0]);
                // y_mode against VP9_KF_Y_MODE_PROB[Dc][Dc] (above=Dc, left=Dc since neighbours absent)
                let y_probs = probs_for_above_left(IntraMode::Dc, IntraMode::Dc);
                let got_y = read_intra_mode_via_tree(&mut r, &y_probs);
                assert_eq!(got_y, y_mode, "Y mode parse for ({y_mode:?}, {uv_mode:?})");
                // uv_mode against VP9_KF_UV_MODE_PROB[got_y]
                let uv_probs = VP9_KF_UV_MODE_PROB[got_y as usize];
                let got_uv = read_intra_mode_via_tree(&mut r, &uv_probs);
                assert_eq!(got_uv, uv_mode, "UV mode parse for ({y_mode:?}, {uv_mode:?})");
            }
        }
    }

    // -----------------------------------------------------------------
    // S4.13 — recursive partition-tree round-trips (HORZ / VERT / SPLIT)
    // -----------------------------------------------------------------

    /// Build a populated `ModeInfo` cell of size `bsize`, intra-DC,
    /// skip=true, segmentation-off.
    fn make_dc_cell(bsize: BlockSize) -> ModeInfo {
        ModeInfo {
            block_size: bsize,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    /// Splat `mi` across the `[mi_row..+y_mis] x [mi_col..+x_mis]`
    /// patch of the frame mi-grid. Mirrors `build_frame_mi_grid`'s
    /// decoder-side splat: every 8x8 cell inside the leaf's coverage
    /// carries the same `ModeInfo`.
    fn splat_cell(
        grid: &mut [Option<ModeInfo>],
        mi_cols: usize,
        mi_row: usize,
        mi_col: usize,
        mi: ModeInfo,
    ) {
        let x_mis = usize::from(mi.block_size.n_8x8_wide());
        let y_mis = usize::from(mi.block_size.n_8x8_high());
        for dr in 0..y_mis {
            for dc in 0..x_mis {
                grid[(mi_row + dr) * mi_cols + (mi_col + dc)] = Some(mi);
            }
        }
    }

    /// Reader-side replay of the writer's partition-tree walk. Drives
    /// `read_partition` / `read_skip` / `read_intra_mode` recursively
    /// against the same partition + neighbour contexts the writer
    /// builds, and accumulates a `Vec<(mi_row, mi_col, BlockSize)>` of
    /// visited leaves for verification.
    #[allow(clippy::too_many_arguments)]
    fn replay_partition_tree(
        r: &mut BoolReader<'_>,
        part_ctx: &mut TestPartCtx,
        nbr: &mut NeighbourCtx,
        fc: &FrameContext,
        bounds: TileBounds,
        mi_row: usize,
        mi_col: usize,
        bsize: BlockSize,
        n4x4_l2: u8,
        leaves: &mut Vec<(usize, usize, BlockSize)>,
    ) {
        let n8x8_l2 = n4x4_l2.saturating_sub(1);
        let num_8x8_wh: usize = 1usize << n8x8_l2;
        let hbs = num_8x8_wh >> 1;
        let mi_row_in_tile = mi_row - bounds.mi_row_start;
        let mi_col_in_tile = mi_col - bounds.mi_col_start;
        let bsl = n8x8_l2 as usize;
        let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);
        let probs = &VP9_KF_PARTITION_PROBS[ctx];
        let partition = read_partition(r, probs, ctx, None);
        let sub = subsize(bsize, partition).expect("legal subsize");

        let mut read_leaf = |r: &mut BoolReader<'_>,
                             nbr: &mut NeighbourCtx,
                             mi_row: usize,
                             mi_col: usize,
                             leaf_bsize: BlockSize| {
            let mi_row_in_tile = mi_row - bounds.mi_row_start;
            let mi_col_in_tile = mi_col - bounds.mi_col_start;
            let (above_mode, left_mode, skip_ctx) =
                nbr.neighbours_for(mi_row_in_tile, mi_col_in_tile);
            let skip = read_skip(r, &fc.skip_probs, skip_ctx, None);
            assert!(skip, "skip @ ({mi_row}, {mi_col})");
            let y_probs = &VP9_KF_Y_MODE_PROB[above_mode as usize][left_mode as usize];
            let y_mode = read_intra_mode(r, y_probs);
            let uv_probs = &VP9_KF_UV_MODE_PROB[y_mode as usize];
            let _uv_mode = read_intra_mode(r, uv_probs);
            let x_mis = usize::from(leaf_bsize.n_8x8_wide());
            let y_mis = usize::from(leaf_bsize.n_8x8_high());
            nbr.write_block(mi_row_in_tile, mi_col_in_tile, x_mis, y_mis, y_mode, true);
            leaves.push((mi_row, mi_col, leaf_bsize));
        };

        match partition {
            PartitionType::None => {
                read_leaf(r, nbr, mi_row, mi_col, sub);
            }
            PartitionType::Horz => {
                read_leaf(r, nbr, mi_row, mi_col, sub);
                read_leaf(r, nbr, mi_row + hbs, mi_col, sub);
            }
            PartitionType::Vert => {
                read_leaf(r, nbr, mi_row, mi_col, sub);
                read_leaf(r, nbr, mi_row, mi_col + hbs, sub);
            }
            PartitionType::Split => {
                replay_partition_tree(
                    r, part_ctx, nbr, fc, bounds, mi_row, mi_col, sub, n8x8_l2, leaves,
                );
                replay_partition_tree(
                    r,
                    part_ctx,
                    nbr,
                    fc,
                    bounds,
                    mi_row,
                    mi_col + hbs,
                    sub,
                    n8x8_l2,
                    leaves,
                );
                replay_partition_tree(
                    r,
                    part_ctx,
                    nbr,
                    fc,
                    bounds,
                    mi_row + hbs,
                    mi_col,
                    sub,
                    n8x8_l2,
                    leaves,
                );
                replay_partition_tree(
                    r,
                    part_ctx,
                    nbr,
                    fc,
                    bounds,
                    mi_row + hbs,
                    mi_col + hbs,
                    sub,
                    n8x8_l2,
                    leaves,
                );
            }
        }

        if (bsize as u8) >= (BlockSize::Block8x8 as u8)
            && (bsize == BlockSize::Block8x8 || partition != PartitionType::Split)
        {
            part_ctx.update_with_subsize(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
        }
    }

    /// Encode the given mi-grid through `write_tile_intra_keyframe`,
    /// then replay through `replay_partition_tree` and return the
    /// recovered leaves. Asserts the writer accepts the grid.
    fn encode_then_replay(
        mi_rows: u32,
        mi_cols: u32,
        grid: &[Option<ModeInfo>],
    ) -> Vec<(usize, usize, BlockSize)> {
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let fc = FrameContext::new_default();
        let mut buf = vec![0u8; 8 * 1024];
        let written = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            write_tile_intra_keyframe(&mut bw, grid, mi_rows, mi_cols, bounds, &fc)
                .expect("write_tile_intra_keyframe should accept legal partition tree");
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(written);

        let mut reader = BoolReader::new(&buf).expect("reader init");
        let mut part_ctx = TestPartCtx::new(mi_cols as usize);
        let mut nbr = NeighbourCtx::new(mi_cols as usize);
        let mut leaves = Vec::new();

        let mut mi_row = 0usize;
        while mi_row < mi_rows as usize {
            part_ctx.reset_left();
            nbr.reset_left();
            let mut mi_col = 0usize;
            while mi_col < mi_cols as usize {
                replay_partition_tree(
                    &mut reader,
                    &mut part_ctx,
                    &mut nbr,
                    &fc,
                    bounds,
                    mi_row,
                    mi_col,
                    BlockSize::Block64x64,
                    4,
                    &mut leaves,
                );
                mi_col += MI_BLOCK_SIZE;
            }
            mi_row += MI_BLOCK_SIZE;
        }
        leaves
    }

    /// 64x64 PARTITION_HORZ — two `Block64x32` leaves stacked.
    #[test]
    fn partition_horz_64x64_round_trip() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        splat_cell(&mut grid, mi_cols as usize, 0, 0, make_dc_cell(BlockSize::Block64x32));
        splat_cell(&mut grid, mi_cols as usize, 4, 0, make_dc_cell(BlockSize::Block64x32));

        let leaves = encode_then_replay(mi_rows, mi_cols, &grid);
        assert_eq!(
            leaves,
            vec![
                (0, 0, BlockSize::Block64x32),
                (4, 0, BlockSize::Block64x32),
            ]
        );
    }

    /// 64x64 PARTITION_VERT — two `Block32x64` leaves side-by-side.
    #[test]
    fn partition_vert_64x64_round_trip() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        splat_cell(&mut grid, mi_cols as usize, 0, 0, make_dc_cell(BlockSize::Block32x64));
        splat_cell(&mut grid, mi_cols as usize, 0, 4, make_dc_cell(BlockSize::Block32x64));

        let leaves = encode_then_replay(mi_rows, mi_cols, &grid);
        assert_eq!(
            leaves,
            vec![
                (0, 0, BlockSize::Block32x64),
                (0, 4, BlockSize::Block32x64),
            ]
        );
    }

    /// 64x64 PARTITION_SPLIT — four `Block32x32` leaves at NONE each.
    #[test]
    fn partition_split_64x64_round_trip() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = make_dc_cell(BlockSize::Block32x32);
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 0, 4, cell);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 4, 4, cell);

        let leaves = encode_then_replay(mi_rows, mi_cols, &grid);
        assert_eq!(
            leaves,
            vec![
                (0, 0, BlockSize::Block32x32),
                (0, 4, BlockSize::Block32x32),
                (4, 0, BlockSize::Block32x32),
                (4, 4, BlockSize::Block32x32),
            ]
        );
    }

    /// 64x64 SPLIT, first quadrant SPLIT into 4x `Block16x16` NONE,
    /// other three quadrants each `Block32x32` NONE.
    #[test]
    fn partition_split_then_split_round_trip() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // First quadrant (0..4, 0..4) split into four 16x16 (mi-size 2).
        let cell16 = make_dc_cell(BlockSize::Block16x16);
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell16);
        splat_cell(&mut grid, mi_cols as usize, 0, 2, cell16);
        splat_cell(&mut grid, mi_cols as usize, 2, 0, cell16);
        splat_cell(&mut grid, mi_cols as usize, 2, 2, cell16);
        // Other three quadrants: 32x32 NONE.
        let cell32 = make_dc_cell(BlockSize::Block32x32);
        splat_cell(&mut grid, mi_cols as usize, 0, 4, cell32);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, cell32);
        splat_cell(&mut grid, mi_cols as usize, 4, 4, cell32);

        let leaves = encode_then_replay(mi_rows, mi_cols, &grid);
        assert_eq!(
            leaves,
            vec![
                // First quadrant (split into 16x16 quartet)
                (0, 0, BlockSize::Block16x16),
                (0, 2, BlockSize::Block16x16),
                (2, 0, BlockSize::Block16x16),
                (2, 2, BlockSize::Block16x16),
                // Remaining three 32x32 quadrants (decoder order: TR, BL, BR)
                (0, 4, BlockSize::Block32x32),
                (4, 0, BlockSize::Block32x32),
                (4, 4, BlockSize::Block32x32),
            ]
        );
    }

    /// Full recursion down to `Block8x8`: 64 -> SPLIT -> 32 ->
    /// SPLIT -> 16 -> SPLIT -> 8 NONE.
    #[test]
    fn partition_full_recursion_to_8x8() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // Top-left 32x32: split into 16x16 quartet. First 16x16 itself
        // split into four 8x8 leaves. Other three 16x16 are NONE.
        let cell8 = make_dc_cell(BlockSize::Block8x8);
        let cell16 = make_dc_cell(BlockSize::Block16x16);
        let cell32 = make_dc_cell(BlockSize::Block32x32);
        // Block (0,0)..(2,2) = top-left 16x16, split into four 8x8.
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell8);
        splat_cell(&mut grid, mi_cols as usize, 0, 1, cell8);
        splat_cell(&mut grid, mi_cols as usize, 1, 0, cell8);
        splat_cell(&mut grid, mi_cols as usize, 1, 1, cell8);
        // The three remaining 16x16 inside the top-left 32x32.
        splat_cell(&mut grid, mi_cols as usize, 0, 2, cell16);
        splat_cell(&mut grid, mi_cols as usize, 2, 0, cell16);
        splat_cell(&mut grid, mi_cols as usize, 2, 2, cell16);
        // The remaining three 32x32 inside the 64x64.
        splat_cell(&mut grid, mi_cols as usize, 0, 4, cell32);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, cell32);
        splat_cell(&mut grid, mi_cols as usize, 4, 4, cell32);

        let leaves = encode_then_replay(mi_rows, mi_cols, &grid);
        assert_eq!(
            leaves,
            vec![
                // Deepest: four 8x8 inside the top-left 16x16
                (0, 0, BlockSize::Block8x8),
                (0, 1, BlockSize::Block8x8),
                (1, 0, BlockSize::Block8x8),
                (1, 1, BlockSize::Block8x8),
                // Remaining three 16x16 in the top-left 32x32
                (0, 2, BlockSize::Block16x16),
                (2, 0, BlockSize::Block16x16),
                (2, 2, BlockSize::Block16x16),
                // Remaining three 32x32 in the 64x64
                (0, 4, BlockSize::Block32x32),
                (4, 0, BlockSize::Block32x32),
                (4, 4, BlockSize::Block32x32),
            ]
        );
    }

    /// `infer_partition` returns `None` when the leaf at the top-left
    /// cell matches the parent `bsize`.
    #[test]
    fn infer_partition_none() {
        let mi_cols = 8usize;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; mi_cols * mi_cols];
        splat_cell(&mut grid, mi_cols, 0, 0, make_dc_cell(BlockSize::Block64x64));
        let p = infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block64x64)
            .expect("legal NONE");
        assert_eq!(p, PartitionType::None);
    }

    /// `infer_partition` returns `Horz` when the leaf at the top-left
    /// matches `subsize(bsize, HORZ)`.
    #[test]
    fn infer_partition_horz() {
        let mi_cols = 8usize;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; mi_cols * mi_cols];
        splat_cell(&mut grid, mi_cols, 0, 0, make_dc_cell(BlockSize::Block64x32));
        let p = infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block64x64)
            .expect("legal HORZ");
        assert_eq!(p, PartitionType::Horz);
    }

    /// `infer_partition` returns `Vert` when the leaf at the top-left
    /// matches `subsize(bsize, VERT)`.
    #[test]
    fn infer_partition_vert() {
        let mi_cols = 8usize;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; mi_cols * mi_cols];
        splat_cell(&mut grid, mi_cols, 0, 0, make_dc_cell(BlockSize::Block32x64));
        let p = infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block64x64)
            .expect("legal VERT");
        assert_eq!(p, PartitionType::Vert);
    }

    /// `infer_partition` returns `Split` when the leaf is smaller
    /// than any of `bsize` / `subsize(HORZ)` / `subsize(VERT)`. The
    /// recursion validates the sub-tree on the next call.
    #[test]
    fn infer_partition_split() {
        let mi_cols = 8usize;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; mi_cols * mi_cols];
        splat_cell(&mut grid, mi_cols, 0, 0, make_dc_cell(BlockSize::Block32x32));
        let p = infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block64x64)
            .expect("legal SPLIT");
        assert_eq!(p, PartitionType::Split);
    }

    /// `infer_partition` rejects a BLOCK_8x8 parent whose top-left
    /// leaf is anything other than `Block8x8` — scope reduction
    /// disallows sub-8 partitioning.
    #[test]
    fn infer_partition_rejects_sub8_partitioning() {
        let mi_cols = 8usize;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; mi_cols * mi_cols];
        // Synthesize a `Block4x4` leaf at (0,0) under a Block8x8 parent.
        grid[0] = Some(make_dc_cell(BlockSize::Block4x4));
        let err = infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block8x8).unwrap_err();
        assert!(matches!(
            err,
            ModesWriterError::InconsistentPartitionTree {
                parent_bsize: BlockSize::Block8x8,
                ..
            }
        ));
    }

    /// Per-partition-type round-trip: writer emits the bit-sequence
    /// for each `PartitionType`, then `read_partition` recovers the
    /// same value.
    #[test]
    fn write_partition_round_trips_each_type() {
        for p in [
            PartitionType::None,
            PartitionType::Horz,
            PartitionType::Vert,
            PartitionType::Split,
        ] {
            let mut buf = vec![0u8; 64];
            let written = {
                let mut bw = BoolWriter::start_encode(&mut buf);
                // Use ctx=0 — keyframe partition probs slot is fine
                // regardless; the round-trip is symmetric.
                write_partition(&mut bw, &VP9_KF_PARTITION_PROBS[0], p);
                bw.stop_encode().expect("stop_encode")
            };
            buf.truncate(written);
            let mut r = BoolReader::new(&buf).expect("reader");
            let probs = &VP9_KF_PARTITION_PROBS[0];
            let got = read_partition(&mut r, probs, 0, None);
            assert_eq!(got, p, "round-trip for {p:?}");
        }
    }

    /// Inconsistent partition tree: top-left cell at (0, 0) claims
    /// `Block32x64` (implying VERT) but the adjacent cell at (0, 4)
    /// claims `Block16x16` — neither a pure VERT (the right half
    /// would need Block32x64 too) nor a clean SPLIT-then-NONE (the
    /// left half would need Block32x32). The writer rejects.
    #[test]
    fn rejects_inconsistent_partition_tree() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // (0, 0) claims VERT (Block32x64).
        splat_cell(&mut grid, mi_cols as usize, 0, 0, make_dc_cell(BlockSize::Block32x64));
        // (0, 4) claims Block16x16 instead of the Block32x64 a real
        // VERT would require.
        splat_cell(&mut grid, mi_cols as usize, 0, 4, make_dc_cell(BlockSize::Block16x16));
        // Fill the bottom half so the writer doesn't trip on missing
        // cells first.
        splat_cell(&mut grid, mi_cols as usize, 4, 0, make_dc_cell(BlockSize::Block32x32));
        splat_cell(&mut grid, mi_cols as usize, 4, 4, make_dc_cell(BlockSize::Block32x32));

        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let fc = FrameContext::new_default();
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err =
            write_tile_intra_keyframe(&mut bw, &grid, mi_rows, mi_cols, bounds, &fc).unwrap_err();
        // The exact variant depends on where the mismatch surfaces.
        // For this grid: the top-level partition is inferred as VERT
        // (from the (0,0) leaf), then the second VERT leaf at (0,4)
        // carries `block_size = Block16x16` which doesn't match the
        // leaf-bsize the recursion expects (`Block32x64`) — so the
        // writer surfaces `UnsupportedBlockSize` at that cell.
        assert!(
            matches!(err, ModesWriterError::UnsupportedBlockSize { .. })
                || matches!(err, ModesWriterError::InconsistentPartitionTree { .. }),
            "expected UnsupportedBlockSize or InconsistentPartitionTree, got: {err:?}"
        );
    }

    // -----------------------------------------------------------------
    // S4.16c — inter-frame partition walker + per-leaf wire emitter
    // -----------------------------------------------------------------

    use crate::bitstream::{InterpFilter, ReferenceMode};
    use crate::inter::mode_info::{
        get_intra_inter_context, get_mode_context, read_inter_block_mode_info,
        setup_compound_reference_mode, InterModeReadParams,
    };
    use crate::inter::mv_ref::{sign_bias_from_header, BlockBounds, PrevFrameMvRef};
    use crate::mode_info::{read_is_inter, read_tx_size, BmiInfo, InterMode, Mv, RefFrame};
    use crate::transforms::TxMode;

    /// Default frame-level inter params: SINGLE ref mode, no sign-bias,
    /// no HP, fixed 8-tap filter.
    fn s4c_default_params() -> InterModeReadParams {
        let sign_bias = sign_bias_from_header(&[false, false, false]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        InterModeReadParams {
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            reference_mode: ReferenceMode::Single,
            comp_fixed_ref: fixed,
            comp_var_ref: var,
            sign_bias,
        }
    }

    /// Build an inter `ModeInfo` with `skip = true`, `segment_id = 0`.
    fn make_inter_cell(
        bsize: BlockSize,
        inter_mode: InterMode,
        ref_frames: [RefFrame; 2],
        mvs: [Mv; 2],
        interp_filter: InterpFilter,
    ) -> ModeInfo {
        // For inter blocks with `skip = true` and any non-Select
        // tx_mode, the decoder derives `tx_size = min(max_tx_size,
        // cap)`. Pick `max_txsize(bsize)` so the writer's
        // debug_assert is happy regardless of bsize under
        // `TxMode::AllowTx32x32` (cap = 3).
        let tx = max_txsize(bsize);
        ModeInfo {
            block_size: bsize,
            tx_size: tx,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: inter_mode,
                ref_frames,
                mvs,
                interp_filter,
            },
            bmi: [BmiInfo::default(); 4],
        }
    }

    /// Replay the encoded buffer through the per-symbol readers,
    /// walking the same partition + neighbour state the writer
    /// produced. Returns the recovered `Vec<(mi_row, mi_col, ModeInfo)>`
    /// for round-trip verification.
    #[allow(clippy::too_many_arguments)]
    fn replay_inter_partition_tree(
        r: &mut BoolReader<'_>,
        fc: &FrameContext,
        params: &InterModeReadParams,
        tx_mode: TxMode,
        part_ctx: &mut TestPartCtx,
        mi_grid: &[Option<ModeInfo>],
        mi_rows: usize,
        mi_cols: usize,
        bounds: TileBounds,
        mi_row: usize,
        mi_col: usize,
        bsize: BlockSize,
        n4x4_l2: u8,
        prev_mv_grid: Option<&[PrevFrameMvRef]>,
        leaves: &mut Vec<(usize, usize, ModeInfo)>,
    ) {
        let n8x8_l2 = n4x4_l2.saturating_sub(1);
        let num_8x8_wh: usize = 1usize << n8x8_l2;
        let hbs = num_8x8_wh >> 1;
        let mi_row_in_tile = mi_row - bounds.mi_row_start;
        let mi_col_in_tile = mi_col - bounds.mi_col_start;
        let bsl = n8x8_l2 as usize;
        let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);
        let probs = &fc.partition_prob[ctx];
        let partition = read_partition(r, probs, ctx, None);
        let sub = subsize(bsize, partition).expect("legal subsize");

        let read_leaf = |r: &mut BoolReader<'_>,
                         leaves: &mut Vec<(usize, usize, ModeInfo)>,
                         mi_row: usize,
                         mi_col: usize,
                         leaf_bsize: BlockSize| {
            // Build the neighbour lookup the writer used.
            let nb = MiGridNeighbourLookup::new(mi_grid, mi_cols, mi_row, mi_col, mi_rows);
            let above = nb.get(Position { row: -1, col: 0 }).copied();
            let left = nb.get(Position { row: 0, col: -1 }).copied();
            let skip_ctx = usize::from(above.is_some_and(|m| m.skip))
                + usize::from(left.is_some_and(|m| m.skip));
            let skip = read_skip(r, &fc.skip_probs, skip_ctx, None);
            assert!(skip, "skip @ ({mi_row}, {mi_col})");
            let intra_inter_ctx = get_intra_inter_context(above.as_ref(), left.as_ref());
            let is_inter = read_is_inter(r, &fc.intra_inter_prob, intra_inter_ctx, None);
            assert!(is_inter, "is_inter @ ({mi_row}, {mi_col})");
            // tx_size — under v1 invariants `allow_select = false`, so
            // no bits are consumed.
            let allow_select = !skip;
            let max_tx = max_txsize(leaf_bsize);
            let above_tx = above.map(|m| (m.tx_size, m.skip));
            let left_tx = left.map(|m| (m.tx_size, m.skip));
            let tx_ctx = if max_tx >= TxSize::Tx8x8 {
                get_tx_size_context(above_tx, left_tx, max_tx)
            } else {
                0
            };
            let tx_size = read_tx_size(r, &fc.tx_probs, tx_mode, max_tx, allow_select, tx_ctx, None);
            // Decode the mode info via the same per-block reader the
            // decoder uses, threading the same NeighbourLookup adapter.
            let bw_mi = i32::from(leaf_bsize.n_8x8_wide());
            let bh_mi = i32::from(leaf_bsize.n_8x8_high());
            let bounds_block = BlockBounds::new(
                mi_row as i32,
                mi_col as i32,
                mi_rows as i32,
                mi_cols as i32,
                bw_mi,
                bh_mi,
            );
            let prev_mv =
                prev_mv_grid.and_then(|g| g.get(mi_row * mi_cols + mi_col).copied());
            let mut got = ModeInfo {
                block_size: leaf_bsize,
                tx_size,
                skip,
                segment_id: 0,
                seg_id_predicted: false,
                mode: BlockMode::Inter {
                    mode: InterMode::NearestMv,
                    ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                    mvs: [Mv::ZERO; 2],
                    interp_filter: params.interp_filter,
                },
                bmi: [BmiInfo::default(); 4],
            };
            read_inter_block_mode_info(
                r, fc, None, None, params, &nb, leaf_bsize, &bounds_block, prev_mv, &mut got,
            );
            // Avoid an unused-variable lint when callers don't read it.
            let _ = get_mode_context(leaf_bsize, &nb);
            leaves.push((mi_row, mi_col, got));
        };

        match partition {
            PartitionType::None => {
                read_leaf(r, leaves, mi_row, mi_col, sub);
            }
            PartitionType::Horz => {
                read_leaf(r, leaves, mi_row, mi_col, sub);
                read_leaf(r, leaves, mi_row + hbs, mi_col, sub);
            }
            PartitionType::Vert => {
                read_leaf(r, leaves, mi_row, mi_col, sub);
                read_leaf(r, leaves, mi_row, mi_col + hbs, sub);
            }
            PartitionType::Split => {
                replay_inter_partition_tree(
                    r, fc, params, tx_mode, part_ctx, mi_grid, mi_rows, mi_cols, bounds, mi_row,
                    mi_col, sub, n8x8_l2, prev_mv_grid, leaves,
                );
                replay_inter_partition_tree(
                    r,
                    fc,
                    params,
                    tx_mode,
                    part_ctx,
                    mi_grid,
                    mi_rows,
                    mi_cols,
                    bounds,
                    mi_row,
                    mi_col + hbs,
                    sub,
                    n8x8_l2,
                    prev_mv_grid,
                    leaves,
                );
                replay_inter_partition_tree(
                    r,
                    fc,
                    params,
                    tx_mode,
                    part_ctx,
                    mi_grid,
                    mi_rows,
                    mi_cols,
                    bounds,
                    mi_row + hbs,
                    mi_col,
                    sub,
                    n8x8_l2,
                    prev_mv_grid,
                    leaves,
                );
                replay_inter_partition_tree(
                    r,
                    fc,
                    params,
                    tx_mode,
                    part_ctx,
                    mi_grid,
                    mi_rows,
                    mi_cols,
                    bounds,
                    mi_row + hbs,
                    mi_col + hbs,
                    sub,
                    n8x8_l2,
                    prev_mv_grid,
                    leaves,
                );
            }
        }

        if (bsize as u8) >= (BlockSize::Block8x8 as u8)
            && (bsize == BlockSize::Block8x8 || partition != PartitionType::Split)
        {
            part_ctx.update_with_subsize(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
        }
    }

    /// Encode `grid` via `write_tile_inter`, then replay via the
    /// per-symbol readers and return the recovered leaves.
    #[allow(clippy::too_many_arguments)]
    fn encode_then_replay_inter(
        fc: &FrameContext,
        params: &InterModeReadParams,
        tx_mode: TxMode,
        mi_rows: u32,
        mi_cols: u32,
        grid: &[Option<ModeInfo>],
        prev_mv_grid: Option<&[PrevFrameMvRef]>,
    ) -> Vec<(usize, usize, ModeInfo)> {
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 16 * 1024];
        let written = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            write_tile_inter(
                &mut bw,
                fc,
                params,
                tx_mode,
                grid,
                mi_rows,
                mi_cols,
                bounds,
                prev_mv_grid,
            )
            .expect("write_tile_inter should accept legal grid");
            // Pad with zero bits so the reader's lookahead is always
            // satisfied.
            for _ in 0..16 {
                bw.write_bit(0);
            }
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(written);

        let mut r = BoolReader::new(&buf).expect("reader init");
        let mut part_ctx = TestPartCtx::new(mi_cols as usize);
        let mut leaves = Vec::new();
        let mut mi_row = 0usize;
        while mi_row < mi_rows as usize {
            part_ctx.reset_left();
            let mut mi_col = 0usize;
            while mi_col < mi_cols as usize {
                replay_inter_partition_tree(
                    &mut r,
                    fc,
                    params,
                    tx_mode,
                    &mut part_ctx,
                    grid,
                    mi_rows as usize,
                    mi_cols as usize,
                    bounds,
                    mi_row,
                    mi_col,
                    BlockSize::Block64x64,
                    4,
                    prev_mv_grid,
                    &mut leaves,
                );
                mi_col += MI_BLOCK_SIZE;
            }
            mi_row += MI_BLOCK_SIZE;
        }
        leaves
    }

    /// Assert recovered `mi`'s inter payload matches `expected`'s.
    fn assert_inter_match(expected: &ModeInfo, got: &ModeInfo, where_: &str) {
        match (&expected.mode, &got.mode) {
            (
                BlockMode::Inter {
                    mode: m_in,
                    ref_frames: rf_in,
                    mvs: mvs_in,
                    interp_filter: if_in,
                },
                BlockMode::Inter {
                    mode: m_out,
                    ref_frames: rf_out,
                    mvs: mvs_out,
                    interp_filter: if_out,
                },
            ) => {
                assert_eq!(m_in, m_out, "{where_}: inter_mode");
                assert_eq!(rf_in, rf_out, "{where_}: ref_frames");
                assert_eq!(mvs_in, mvs_out, "{where_}: mvs");
                assert_eq!(if_in, if_out, "{where_}: interp_filter");
            }
            _ => panic!("{where_}: expected Inter on both sides"),
        }
    }

    /// 64×64 PARTITION_NONE, single-ref LASTFRAME, ZEROMV.
    #[test]
    fn s4_16c_round_trip_inter_partition_none_zero_mv() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];

        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 1);
        assert_eq!((leaves[0].0, leaves[0].1), (0, 0));
        assert_inter_match(&cell, &leaves[0].2, "64x64 NONE ZEROMV");
    }

    /// 64×64 SPLIT into 4× 32×32 PARTITION_NONE, mixed single-ref
    /// modes (LAST/GOLDEN, NEAREST/ZERO).
    #[test]
    fn s4_16c_round_trip_inter_partition_split() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];

        // Top-left: LAST, NEAREST. With no neighbours, find_mv_refs
        // returns count=0 → best_ref_mv = (0,0) → mvs[0] = (0,0).
        let cell_tl = make_inter_cell(
            BlockSize::Block32x32,
            InterMode::NearestMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // Top-right: GOLDEN, ZEROMV.
        let cell_tr = make_inter_cell(
            BlockSize::Block32x32,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // Bottom-left: LAST, ZEROMV.
        let cell_bl = make_inter_cell(
            BlockSize::Block32x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // Bottom-right: GOLDEN, NEAREST.
        let cell_br = make_inter_cell(
            BlockSize::Block32x32,
            InterMode::NearestMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell_tl);
        splat_cell(&mut grid, mi_cols as usize, 0, 4, cell_tr);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, cell_bl);
        splat_cell(&mut grid, mi_cols as usize, 4, 4, cell_br);

        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 4, "4 quadrants");
        let expected = [
            (0usize, 0usize, &cell_tl),
            (0, 4, &cell_tr),
            (4, 0, &cell_bl),
            (4, 4, &cell_br),
        ];
        for (i, (er, ec, ecell)) in expected.iter().enumerate() {
            assert_eq!((leaves[i].0, leaves[i].1), (*er, *ec));
            assert_inter_match(ecell, &leaves[i].2, "split quadrant");
        }
    }

    /// 64×64 HORZ into 2× 64×32, single-ref LAST ZEROMV.
    #[test]
    fn s4_16c_round_trip_inter_horz() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, cell);

        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 2, "2 HORZ leaves");
        assert_eq!((leaves[0].0, leaves[0].1), (0, 0));
        assert_eq!((leaves[1].0, leaves[1].1), (4, 0));
        assert_inter_match(&cell, &leaves[0].2, "horz top");
        assert_inter_match(&cell, &leaves[1].2, "horz bottom");
    }

    /// 64×64 VERT into 2× 32×64, single-ref LAST ZEROMV.
    #[test]
    fn s4_16c_round_trip_inter_vert() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block32x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 0, 4, cell);

        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 2, "2 VERT leaves");
        assert_eq!((leaves[0].0, leaves[0].1), (0, 0));
        assert_eq!((leaves[1].0, leaves[1].1), (0, 4));
        assert_inter_match(&cell, &leaves[0].2, "vert left");
        assert_inter_match(&cell, &leaves[1].2, "vert right");
    }

    /// 64×64 PARTITION_NONE, single-ref LASTFRAME, NEWMV with non-
    /// zero MV (allow_hp=true). With empty neighbours, find_mv_refs
    /// returns count=0 → best_ref_mv = (0,0) → diff == MV itself.
    #[test]
    fn s4_16c_round_trip_inter_new_mv() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.allow_high_precision_mv = true;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mvs = [Mv { row: 12, col: -4 }, Mv::ZERO];
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            mvs,
            InterpFilter::EightTap,
        );
        let grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];

        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 1);
        assert_inter_match(&cell, &leaves[0].2, "NEWMV 64x64");
    }

    /// Leaf carrying `BlockMode::Intra` → writer rejects with
    /// `UnsupportedIntraInInter`.
    #[test]
    fn s4_16c_rejects_intra_leaf() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.mode = BlockMode::Intra {
            y_mode: IntraMode::Dc,
            uv_mode: IntraMode::Dc,
        };
        let grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, ModesWriterError::UnsupportedIntraInInter { .. }),
            "got: {err:?}"
        );
    }

    /// Leaf with `skip = false` → writer rejects with
    /// `UnsupportedNonSkip`.
    #[test]
    fn s4_16c_rejects_non_skip_leaf() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.skip = false;
        let grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, ModesWriterError::UnsupportedNonSkip { .. }),
            "got: {err:?}"
        );
    }

    /// Leaf with `segment_id != 0` → writer rejects with
    /// `UnsupportedSegmentation`.
    #[test]
    fn s4_16c_rejects_nonzero_segment_id() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.segment_id = 1;
        let grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, ModesWriterError::UnsupportedSegmentation { .. }),
            "got: {err:?}"
        );
    }

    /// Smallest reachable inter-frame partition. The walker stops
    /// recursion at BLOCK_8x8 (forcing PARTITION_NONE there), so an
    /// 8x8-grain SB is the densest leaf the partition writer
    /// produces today. Sub-8 inter leaves (BLOCK_4x4/4x8/8x4) are
    /// reached only through the inter-block writer's own sub-8
    /// path (S4.16e) and aren't exercised by this walker test;
    /// see the dedicated `s4_16e_*` tests in `inter::mode_info`.
    #[test]
    fn s4_16c_round_trip_8x8_grain_dense_sb() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        for r in 0..8usize {
            for c in 0..8usize {
                splat_cell(&mut grid, mi_cols as usize, r, c, cell);
            }
        }
        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        // 64 Block8x8 leaves in scan order. Just spot-check the count.
        assert_eq!(leaves.len(), 64, "8x8 = 64 leaves");
        // Spot-check one.
        assert_inter_match(&cell, &leaves[0].2, "8x8 leaf [0]");
    }

    /// `write_tile_inter` rejects an empty cell.
    #[test]
    fn s4_16c_rejects_missing_cell() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, ModesWriterError::MissingCell { mi_row: 0, mi_col: 0 }),
            "got: {err:?}"
        );
    }

    /// `write_tile_inter` rejects non-multiple-of-64 dims.
    #[test]
    fn s4_16c_rejects_non_multiple_of_sb() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 7u32;
        let mi_cols = 8u32;
        let grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 7,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, ModesWriterError::NonMultipleOfSb { .. }), "got: {err:?}");
    }
}
