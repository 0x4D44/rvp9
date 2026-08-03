//! Per-tile mode-info writer (keyframe partition-tree scaffold).
//!
//! Encoder mirror of the decoder's per-tile partition-tree walker
//! (`rvp9_decoder::tile::decode_tile_intra` + `decode_partition` +
//! `decode_block`). v1 scope was every-SB-PARTITION_NONE; S4.13
//! widens the writer to mirror the decoder's full recursive
//! partition tree (HORZ / VERT / SPLIT down to BLOCK_8x8 leaves),
//! with real per-leaf intra modes and residuals.
//!
//! Each leaf block is still required to be intra (any of the 10
//! Y/UV modes); segmentation must be disabled. A leaf may set skip
//! when every luma and chroma transform has EOB zero.
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
//! Partial superblocks use the same reduced partition syntax as the
//! decoder: an unavailable lower or right half removes impossible
//! partition choices, and an entirely outside recursive quadrant is
//! skipped. Product encoder widths remain superblock-aligned, while
//! heights may end on any 8-pixel mi-row boundary.
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
//!    - **Skip flag** — encoded against `fc.skip_probs[skip_ctx]`.
//!    - **Y intra mode** — chosen mode encoded as the
//!      `VP9_INTRA_MODE_TREE` walk against
//!      `VP9_KF_Y_MODE_PROB[above_mode][left_mode]`.
//!    - **UV intra mode** — chosen mode encoded against
//!      `VP9_KF_UV_MODE_PROB[y_mode]`.
//!    - **tx_size** — NOT emitted on the wire (caller's
//!      `tx_mode != TxModeSelect`).
//!    - **Residual tokens** — emitted for non-skip leaves; skip leaves
//!      reset the coefficient contexts exactly as the decoder does.
//!
//! ## Caller contract
//!
//! - The compressed header must commit
//!   `tx_mode = AllowTx32x32` (or any non-`TxModeSelect` mode whose
//!   `tx_mode_to_biggest_tx_size` matches each leaf's expected
//!   `tx_size`). The wire format cannot otherwise communicate the
//!   per-block tx-size when it is implicit.
//! - The frame must be a keyframe (or intra-only) and segmentation
//!   must be disabled in the uncompressed header.
//! - Frame dimensions must be multiples of 64 pixels (= 8 mi-units).
//!   Partial superblocks at the frame edge are future work.

use super::error::BitstreamError;
use super::inter_coef_source::InterCoefSource;
use super::intra_coef_source::IntraCoefSource;
use crate::bitstream::SEG_TREE_PROBS;
use crate::bitstream::compressed_header::tx_mode_to_biggest_tx_size;
use crate::block::{BlockSize, PartitionType, max_txsize, subsize};
use crate::coef_ctx::TileCoefCtx;
use crate::entropy::adapt::CoefCountsBank;
use crate::entropy::bitacct::{BitCategory, set_category};
use crate::entropy::{
    BoolWriter, CoefTokenPass, FrameContext, FrameCounts, RetainedCoefTokens,
    VP9_KF_PARTITION_PROBS, VP9_KF_UV_MODE_PROB, VP9_KF_Y_MODE_PROB, write_block_tokens_i16,
    write_block_tokens_with_retained_i16,
};
use crate::inter::mode_info::{
    InterModeReadParams, InterModeWriteError, get_intra_inter_context, write_inter_block_mode_info,
};
use crate::inter::mv_ref::{BlockBounds, NeighbourLookup, Position, PrevFrameMvRef};
use crate::intra_pred::IntraMode;
use crate::mode_info::{BlockMode, BmiInfo, ModeInfo, get_tx_size_context, write_segment_id};
use crate::transforms::{TxMode, TxSize, TxType};

/// Encoder-side segmentation context for the per-block writer.
///
/// AQ.0 scope: the writer emits the per-block `segment_id` only when
/// `Some(SegmentationWriteParams)` is supplied. This matches libvpx's
/// cyclic-refresh AQ (mode 3) which runs with `update_map = true &&
/// temporal_update = false`. The `temporal_update == true` path
/// (predicted-segment-id flag bit against `seg.pred_probs`) is
/// deferred — libvpx's cyclic refresh doesn't use it, and a v1 writer
/// for the simple-tree-decode case covers AQ.3's needs end-to-end.
///
/// The decoder mirror is the per-block branch in
/// `rvp9_decoder::tile::decode_block` (`segmentation_update_map`
/// gate) for keyframes and `read_inter_segment_id` (with
/// `update_map && !temporal_update`) for inter frames.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SegmentationWriteParams<'a> {
    /// Tree probabilities for the segment-id tree decode. Mirrors
    /// `SegmentationParams::tree_probs` (the header-emitted slice the
    /// decoder reads back into `params.seg_tree_probs`).
    pub tree_probs: &'a [u8; SEG_TREE_PROBS],
}

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

/// Return one VP9 tile edge in 8×8 mode-info units.
///
/// Mirrors libvpx `get_tile_offset` in `vp9_tile_common.c`: tile
/// boundaries divide the superblock count, then convert back to MI
/// units. This is deliberately not pixel- or MI-width division when
/// the number of 64×64 columns is not divisible by the tile count.
///
/// # Panics
///
/// Panics when `log2_tile_dim` exceeds VP9's maximum of six.
#[inline]
#[must_use]
pub fn tile_mi_offset(idx: u32, mi_dim: usize, log2_tile_dim: u8) -> usize {
    assert!(
        log2_tile_dim <= 6,
        "VP9 log2 tile dimension must be in 0..=6"
    );
    let sb_dim = mi_dim.div_ceil(MI_BLOCK_SIZE);
    let raw = ((idx as usize) * sb_dim) >> log2_tile_dim;
    (raw * MI_BLOCK_SIZE).min(mi_dim)
}

/// Build the MI bounds for one VP9 tile.
///
/// Tile rows are included for bitstream completeness. The v1
/// real-time encoder currently requests column tiles only.
#[must_use]
pub fn tile_bounds(
    mi_rows: usize,
    mi_cols: usize,
    log2_tile_rows: u8,
    log2_tile_cols: u8,
    tile_row: u32,
    tile_col: u32,
) -> TileBounds {
    TileBounds {
        mi_row_start: tile_mi_offset(tile_row, mi_rows, log2_tile_rows),
        mi_row_end: tile_mi_offset(tile_row + 1, mi_rows, log2_tile_rows),
        mi_col_start: tile_mi_offset(tile_col, mi_cols, log2_tile_cols),
        mi_col_end: tile_mi_offset(tile_col + 1, mi_cols, log2_tile_cols),
    }
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
    /// A cell carried `skip == false` in a writer scope that doesn't
    /// support residual emission. The keyframe (intra) writer still
    /// requires `skip == true` for every leaf; the inter writer now
    /// supports `skip == false` (S4.16g) and never returns this.
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
    /// Legacy geometry error retained for API compatibility. Current
    /// writers support partial frame-edge superblocks.
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
    /// Retained coefficient blocks did not match the final syntax walk.
    RetainedCoefMismatch,
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
                "modes_writer v1: cell ({mi_row}, {mi_col}) has unsupported or inconsistent intra mode info"
            ),
            Self::UnsupportedNonSkip { mi_row, mi_col } => write!(
                f,
                "modes_writer v1: cell ({mi_row}, {mi_col}) has skip=false; this writer scope emits no residual tokens"
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
            Self::RetainedCoefMismatch => {
                f.write_str("modes_writer: retained coefficient tokens did not match tile syntax")
            }
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
/// - The compressed header must commit the same `tx_mode` passed here.
///   Under `TxModeSelect` (Track-F iter8) each leaf's RD-chosen
///   `tx_size` is signalled on the wire via [`write_tx_size`]; under any
///   non-select mode no tx_size bits are emitted and the decoder derives
///   `tx_size = min(max_tx, tx_mode_to_biggest_tx_size(tx_mode))`.
/// - The uncompressed header must commit `segmentation.enabled =
///   false`.
/// - The frame must be a keyframe (or intra-only), with dimensions on
///   8-pixel mi-row boundaries.
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
/// - `InvalidTileBounds` — geometry-level rejection.
/// - [`ModesWriterError::BufferOverflow`] — the [`BoolWriter`]
///   overflowed.
#[allow(clippy::too_many_arguments)] // mirrors libvpx's per-tile signature
pub fn write_tile_intra_keyframe(
    bw: &mut BoolWriter<'_>,
    mi_grid: &[Option<ModeInfo>],
    mi_rows: u32,
    mi_cols: u32,
    tile_bounds: TileBounds,
    fc: &FrameContext,
    tx_mode: TxMode,
    seg: Option<SegmentationWriteParams<'_>>,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn IntraCoefSource,
) -> Result<(), ModesWriterError> {
    write_tile_intra_keyframe_with_counts(
        bw,
        mi_grid,
        mi_rows,
        mi_cols,
        tile_bounds,
        fc,
        tx_mode,
        seg,
        coef_ctx,
        coef_source,
        None,
    )
}

/// [`write_tile_intra_keyframe`] with optional frame counts for the
/// encoder's pre-header coefficient and skip probability-update pass.
#[allow(clippy::too_many_arguments)]
pub fn write_tile_intra_keyframe_with_counts(
    bw: &mut BoolWriter<'_>,
    mi_grid: &[Option<ModeInfo>],
    mi_rows: u32,
    mi_cols: u32,
    tile_bounds: TileBounds,
    fc: &FrameContext,
    tx_mode: TxMode,
    seg: Option<SegmentationWriteParams<'_>>,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn IntraCoefSource,
    mut counts: Option<&mut FrameCounts>,
) -> Result<(), ModesWriterError> {
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
        // SB-row boundary: clear the per-plane left coef-context. The
        // above row persists across SB rows. Mirrors libvpx
        // `vp9_decodeframe.c:2089` (same gate the inter writer applies
        // in `write_tile_inter`).
        coef_ctx.reset_left_coef();

        let mut mi_col = tile_bounds.mi_col_start;
        while mi_col < tile_bounds.mi_col_end {
            write_partition_tree(
                bw,
                &mut part_ctx,
                &mut nbr_ctx,
                &tile_bounds,
                mi_grid,
                mi_rows_us,
                mi_cols_us,
                mi_row,
                mi_col,
                BlockSize::Block64x64,
                4, // n4x4_l2 for 64x64 = log2(16) = 4
                fc,
                tx_mode,
                seg,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
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
/// - `bsize == Block8x8`: match the three legal sub-8 leaves exactly:
///   `Block8x4` → HORZ, `Block4x8` → VERT, and `Block4x4` → SPLIT.
///   The caller handles these through the VP9 `hbs == 0` one-cell
///   branch rather than recursing below the mi-grid resolution.
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

    // BLOCK_8x8 parents terminate the mi-grid walk. Their sub-8
    // partition lives inside one ModeInfo cell and is represented by
    // its `bmi` array, so match the leaf size without recursing.
    if bsize == BlockSize::Block8x8 {
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
        if let Some(split) = subsize(bsize, PartitionType::Split)
            && leaf_bsize == split
        {
            return Ok(PartitionType::Split);
        }
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

/// Inter-aware variant of [`infer_partition`]. Same as the keyframe
/// version for every parent size **except** `Block8x8`, where it
/// recognises sub-8 leaves (`Block8x4` / `Block4x8` / `Block4x4`)
/// as `PARTITION_HORZ` / `PARTITION_VERT` / `PARTITION_SPLIT`
/// respectively. The decoder's `decode_inter_partition` (`tile.rs`)
/// reaches this point via the `hbs == 0` branch and calls
/// `decode_inter_block` once with `sub = subsize(Block8x8,
/// partition)` — the sub-partitioning is then handled internally by
/// `resolve_sub8_bmi` (mirrored by S4.16e's `write_sub8_bmi`).
///
/// Keyframes call [`infer_partition`], which now recognises the same
/// legal sub-8 partition shapes and writes keyframe BMI modes.
fn infer_inter_partition(
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

    // BLOCK_8x8 parents on the inter path can split into 4x4 / 4x8 /
    // 8x4 sub-8 leaves. Match against `subsize(BLOCK_8x8, *)`.
    if bsize == BlockSize::Block8x8 {
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
        if let Some(split) = subsize(bsize, PartitionType::Split)
            && leaf_bsize == split
        {
            return Ok(PartitionType::Split);
        }
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
fn write_partition(bw: &mut BoolWriter<'_>, probs: &[u8; 3], partition: PartitionType) {
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
/// At frame edges, unavailable halves use the decoder's reduced
/// partition syntax and outside recursive quadrants return immediately.
/// At `hbs == 0`, the writer mirrors the decoder's one-call branch: the
/// sub-8 leaf and its four BMI slots occupy one 8x8 mi-grid cell.
#[allow(clippy::too_many_arguments)] // mirrors libvpx 1:1
fn write_partition_tree(
    bw: &mut BoolWriter<'_>,
    part_ctx: &mut PartitionCtx,
    nbr_ctx: &mut NeighbourCtx,
    bounds: &TileBounds,
    mi_grid: &[Option<ModeInfo>],
    mi_rows: usize,
    mi_cols: usize,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    n4x4_l2: u8,
    fc: &FrameContext,
    tx_mode: TxMode,
    seg: Option<SegmentationWriteParams<'_>>,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn IntraCoefSource,
    mut counts: Option<&mut FrameCounts>,
) -> Result<(), ModesWriterError> {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return Ok(());
    }
    debug_assert!(mi_row < bounds.mi_row_end && mi_col < bounds.mi_col_end);

    let n8x8_l2: u8 = n4x4_l2.saturating_sub(1);
    let num_8x8_wh: usize = 1usize << n8x8_l2;
    let hbs: usize = num_8x8_wh >> 1;

    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let bsl: usize = n8x8_l2 as usize;
    let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);

    let has_rows = mi_row + hbs < mi_rows;
    let has_cols = mi_col + hbs < mi_cols;
    let partition = infer_partition(mi_grid, mi_cols, mi_row, mi_col, bsize)?;
    let probs = &VP9_KF_PARTITION_PROBS[ctx];
    if has_rows && has_cols {
        write_partition(bw, probs, partition);
    } else if !has_rows && has_cols {
        debug_assert!(matches!(
            partition,
            PartitionType::Horz | PartitionType::Split
        ));
        bw.write(u8::from(partition == PartitionType::Split), probs[1]);
    } else if has_rows && !has_cols {
        debug_assert!(matches!(
            partition,
            PartitionType::Vert | PartitionType::Split
        ));
        bw.write(u8::from(partition == PartitionType::Split), probs[2]);
    } else {
        debug_assert_eq!(partition, PartitionType::Split);
    }

    let sub = subsize(bsize, partition)
        .expect("infer_partition only returns a partition whose subsize is legal");

    if hbs == 0 {
        let mi = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
            .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
        write_block_intra_keyframe(
            bw,
            fc,
            tx_mode,
            seg,
            nbr_ctx,
            bounds,
            mi_grid,
            mi_rows,
            mi_cols,
            mi_row,
            mi_col,
            sub,
            &mi,
            coef_ctx,
            coef_source,
            counts.as_deref_mut(),
        )?;
        part_ctx.update(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
        return Ok(());
    }

    match partition {
        PartitionType::None => {
            // sub == bsize for PARTITION_NONE. Write the leaf.
            let mi = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_block_intra_keyframe(
                bw,
                fc,
                tx_mode,
                seg,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row,
                mi_col,
                sub,
                &mi,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
            )?;
        }
        PartitionType::Horz => {
            // Two leaves: (mi_row, mi_col) and (mi_row + hbs, mi_col),
            // both with `sub` = horizontal subsize of `bsize`.
            let top = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_block_intra_keyframe(
                bw,
                fc,
                tx_mode,
                seg,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row,
                mi_col,
                sub,
                &top,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
            )?;
            if has_rows {
                let mi_row_b = mi_row + hbs;
                let bot = mi_grid_get(mi_grid, mi_cols, mi_row_b, mi_col).ok_or(
                    ModesWriterError::MissingCell {
                        mi_row: mi_row_b,
                        mi_col,
                    },
                )?;
                write_block_intra_keyframe(
                    bw,
                    fc,
                    tx_mode,
                    seg,
                    nbr_ctx,
                    bounds,
                    mi_grid,
                    mi_rows,
                    mi_cols,
                    mi_row_b,
                    mi_col,
                    sub,
                    &bot,
                    coef_ctx,
                    coef_source,
                    counts.as_deref_mut(),
                )?;
            }
        }
        PartitionType::Vert => {
            let left = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_block_intra_keyframe(
                bw,
                fc,
                tx_mode,
                seg,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row,
                mi_col,
                sub,
                &left,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
            )?;
            if has_cols {
                let mi_col_r = mi_col + hbs;
                let right = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col_r).ok_or(
                    ModesWriterError::MissingCell {
                        mi_row,
                        mi_col: mi_col_r,
                    },
                )?;
                write_block_intra_keyframe(
                    bw,
                    fc,
                    tx_mode,
                    seg,
                    nbr_ctx,
                    bounds,
                    mi_grid,
                    mi_rows,
                    mi_cols,
                    mi_row,
                    mi_col_r,
                    sub,
                    &right,
                    coef_ctx,
                    coef_source,
                    counts.as_deref_mut(),
                )?;
            }
        }
        PartitionType::Split => {
            // 4-way recursion in (mi_row | +hbs, mi_col | +hbs) order
            // — same order as the decoder.
            write_partition_tree(
                bw,
                part_ctx,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row,
                mi_col,
                sub,
                n8x8_l2,
                fc,
                tx_mode,
                seg,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
            )?;
            write_partition_tree(
                bw,
                part_ctx,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row,
                mi_col + hbs,
                sub,
                n8x8_l2,
                fc,
                tx_mode,
                seg,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
            )?;
            write_partition_tree(
                bw,
                part_ctx,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row + hbs,
                mi_col,
                sub,
                n8x8_l2,
                fc,
                tx_mode,
                seg,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
            )?;
            write_partition_tree(
                bw,
                part_ctx,
                nbr_ctx,
                bounds,
                mi_grid,
                mi_rows,
                mi_cols,
                mi_row + hbs,
                mi_col + hbs,
                sub,
                n8x8_l2,
                fc,
                tx_mode,
                seg,
                coef_ctx,
                coef_source,
                counts,
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

#[inline]
fn keyframe_neighbour_y_mode(neighbour: Option<&ModeInfo>, block: usize) -> IntraMode {
    let Some(neighbour) = neighbour else {
        return IntraMode::Dc;
    };
    let BlockMode::Intra { y_mode, .. } = neighbour.mode else {
        return IntraMode::Dc;
    };
    if neighbour.block_size < BlockSize::Block8x8 {
        IntraMode::from_u8(neighbour.bmi[block].as_mode).unwrap_or(y_mode)
    } else {
        y_mode
    }
}

#[inline]
fn keyframe_above_block_mode(
    block: usize,
    bmi: &[BmiInfo; 4],
    above: Option<&ModeInfo>,
) -> IntraMode {
    if block < 2 {
        keyframe_neighbour_y_mode(above, block + 2)
    } else {
        IntraMode::from_u8(bmi[block - 2].as_mode).unwrap_or(IntraMode::Dc)
    }
}

#[inline]
fn keyframe_left_block_mode(
    block: usize,
    bmi: &[BmiInfo; 4],
    left: Option<&ModeInfo>,
) -> IntraMode {
    if block == 0 || block == 2 {
        keyframe_neighbour_y_mode(left, block + 1)
    } else {
        IntraMode::from_u8(bmi[block - 1].as_mode).unwrap_or(IntraMode::Dc)
    }
}

#[inline]
fn keyframe_luma_mode_for_tx(
    block_size: BlockSize,
    bmi: &[BmiInfo; 4],
    parent_y_mode: IntraMode,
    tx_x4_local: usize,
    tx_y4_local: usize,
) -> IntraMode {
    if block_size < BlockSize::Block8x8 {
        let bmi_index = (tx_y4_local << 1) | tx_x4_local;
        IntraMode::from_u8(bmi[bmi_index].as_mode).unwrap_or(parent_y_mode)
    } else {
        parent_y_mode
    }
}

#[allow(clippy::too_many_arguments)]
fn write_sub8_keyframe_y_modes(
    bw: &mut BoolWriter<'_>,
    block_size: BlockSize,
    parent_y_mode: IntraMode,
    bmi: &[BmiInfo; 4],
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
    mi_row: usize,
    mi_col: usize,
) -> Result<(), ModesWriterError> {
    let mode_at = |index: usize| {
        IntraMode::from_u8(bmi[index].as_mode)
            .ok_or(ModesWriterError::UnsupportedBlockMode { mi_row, mi_col })
    };
    let write_one = |bw: &mut BoolWriter<'_>, index: usize, mode: IntraMode| {
        let above_mode = keyframe_above_block_mode(index, bmi, above);
        let left_mode = keyframe_left_block_mode(index, bmi, left);
        write_intra_mode(
            bw,
            &VP9_KF_Y_MODE_PROB[above_mode as usize][left_mode as usize],
            mode,
        );
    };

    match block_size {
        BlockSize::Block4x4 => {
            if parent_y_mode != mode_at(3)? {
                return Err(ModesWriterError::UnsupportedBlockMode { mi_row, mi_col });
            }
            for index in 0..4 {
                write_one(bw, index, mode_at(index)?);
            }
        }
        BlockSize::Block4x8 => {
            let mode0 = mode_at(0)?;
            let mode1 = mode_at(1)?;
            if mode_at(2)? != mode0 || mode_at(3)? != mode1 || parent_y_mode != mode1 {
                return Err(ModesWriterError::UnsupportedBlockMode { mi_row, mi_col });
            }
            write_one(bw, 0, mode0);
            write_one(bw, 1, mode1);
        }
        BlockSize::Block8x4 => {
            let mode0 = mode_at(0)?;
            let mode2 = mode_at(2)?;
            if mode_at(1)? != mode0 || mode_at(3)? != mode2 || parent_y_mode != mode2 {
                return Err(ModesWriterError::UnsupportedBlockMode { mi_row, mi_col });
            }
            write_one(bw, 0, mode0);
            write_one(bw, 2, mode2);
        }
        _ => {
            return Err(ModesWriterError::UnsupportedBlockSize {
                mi_row,
                mi_col,
                block_size,
            });
        }
    }
    Ok(())
}

/// Write the per-leaf keyframe mode info for the block of size
/// `leaf_bsize` starting at `(mi_row, mi_col)`. The leaf is required
/// to be:
///
/// - `block_size == leaf_bsize`,
/// - `mode == Intra { y_mode, uv_mode }` (any of the 10 intra modes),
/// - `segment_id == 0 && !seg_id_predicted`.
///
/// Emits, in order:
///
/// 1. `mi.skip` against `fc.skip_probs[skip_ctx]`.
/// 2. `y_mode` against
///    `VP9_KF_Y_MODE_PROB[above_mode][left_mode]` via the
///    `VP9_INTRA_MODE_TREE` walk.
/// 3. `uv_mode` against `VP9_KF_UV_MODE_PROB[y_mode]` (same tree).
/// 4. When `!mi.skip`, per-plane × per-TX residual coefs via
///    [`write_intra_block_residual`] (drives `write_block_tokens`).
///
/// Then updates `nbr_ctx` with the just-written `(y_mode, skip)`
/// pair so subsequent blocks see the correct above/left neighbour
/// state. Mirrors the relevant slice of the decoder's
/// `TileDecodeCtx::write_block`.
#[allow(clippy::too_many_arguments)] // mirrors the decoder's per-block call
fn write_block_intra_keyframe(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    tx_mode: TxMode,
    seg: Option<SegmentationWriteParams<'_>>,
    nbr_ctx: &mut NeighbourCtx,
    bounds: &TileBounds,
    mi_grid: &[Option<ModeInfo>],
    _mi_rows: usize,
    mi_cols: usize,
    mi_row: usize,
    mi_col: usize,
    leaf_bsize: BlockSize,
    mi: &ModeInfo,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn IntraCoefSource,
    mut counts: Option<&mut FrameCounts>,
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
    // Keyframes: `seg_id_predicted` is decoder-write-only (the inter
    // path stashes the temporal-update pred-flag bit there). Reject
    // any leaf that has it set even when the writer is segment-aware.
    if mi.seg_id_predicted {
        return Err(ModesWriterError::UnsupportedSegmentation { mi_row, mi_col });
    }
    // When the writer scope is "no segmentation", reject any non-zero
    // segment_id — the wire format can't carry it.
    if seg.is_none() && mi.segment_id != 0 {
        return Err(ModesWriterError::UnsupportedSegmentation { mi_row, mi_col });
    }

    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let (_, _, skip_ctx) = nbr_ctx.neighbours_for(mi_row_in_tile, mi_col_in_tile);

    // 0. Segment_id. When `seg.is_some()`, the caller has committed
    //    `segmentation.enabled && segmentation.update_map` in the
    //    uncompressed header, so the decoder reads a tree decode
    //    via `read_segment_id` (mirrors `rvp9_decoder::tile::
    //    decode_block`'s `params.segmentation_update_map` branch).
    //    When `None`, no bits are emitted.
    if let Some(seg) = seg {
        write_segment_id(bw, mi.segment_id, seg.tree_probs);
    }

    // 1. Skip flag against `fc.skip_probs[skip_ctx]`. The skip prob bank
    //    is laid out as `[u8; SKIP_CONTEXTS]` (3 entries) and the
    //    decoder reads with `r.read(probs[ctx])` (`mode_info.rs::
    //    read_skip`).
    bw.write(u8::from(mi.skip), fc.skip_probs[skip_ctx]);
    if let Some(c) = counts.as_deref_mut() {
        c.skip[skip_ctx][usize::from(mi.skip)] += 1;
    }

    // 2. tx_size. The decoder reads tx_size RIGHT AFTER skip and BEFORE
    //    the Y mode (`rvp9_decoder::tile::decode_block`, read_skip ->
    //    read_tx_size -> read_keyframe_y_mode). For intra blocks
    //    `allow_select = !lossless` is always true (the KF is never
    //    lossless here and KF leaves never skip). Under `TxModeSelect`
    //    (Track-F iter8) the per-block RD-chosen `mi.tx_size` is signalled
    //    via the selected-tx-size tree; under any non-select mode
    //    `write_tx_size` emits no bits and the decoder derives the size
    //    from `min(max_tx, tx_mode_to_biggest_tx_size(tx_mode))`. The
    //    tx-size context comes from the above/left neighbours' (tx_size,
    //    skip) — the same `MiGridNeighbourLookup` the inter writer uses.
    let allow_select = !mi.skip; // intra: `!is_inter = true`, so == !skip
    let max_tx = max_txsize(leaf_bsize);
    let tx_neighbours = MiGridNeighbourLookup::new(mi_grid, mi_cols, mi_row, mi_col, *bounds);
    let tx_above = tx_neighbours.get(Position { row: -1, col: 0 });
    let tx_left = tx_neighbours.get(Position { row: 0, col: -1 });
    let tx_ctx = if max_tx >= TxSize::Tx8x8 {
        get_tx_size_context(
            tx_above.map(|m| (m.tx_size, m.skip)),
            tx_left.map(|m| (m.tx_size, m.skip)),
            max_tx,
        )
    } else {
        0
    };
    // Mirror the decoder's `allow_select == false` derivation so the
    // writer accepts any `mi.tx_size` consistent with it; in select mode
    // trust the caller's RD-chosen size.
    let effective_tx_size = if allow_select && tx_mode == TxMode::TxModeSelect {
        mi.tx_size
    } else {
        let cap = tx_mode_to_biggest_tx_size(tx_mode) as u8;
        let raw = core::cmp::min(max_tx as u8, cap);
        match raw {
            0 => TxSize::Tx4x4,
            1 => TxSize::Tx8x8,
            2 => TxSize::Tx16x16,
            _ => TxSize::Tx32x32,
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

    // 3. Intra Y mode(s). A sub-8 ModeInfo carries up to four causal
    //    BMI modes inside its one mi-grid cell. Larger blocks retain
    //    the existing single-mode path.
    if leaf_bsize < BlockSize::Block8x8 {
        write_sub8_keyframe_y_modes(
            bw, leaf_bsize, y_mode, &mi.bmi, tx_above, tx_left, mi_row, mi_col,
        )?;
    } else {
        let above_mode = keyframe_above_block_mode(0, &mi.bmi, tx_above);
        let left_mode = keyframe_left_block_mode(0, &mi.bmi, tx_left);
        let y_probs = &VP9_KF_Y_MODE_PROB[above_mode as usize][left_mode as usize];
        write_intra_mode(bw, y_probs, y_mode);
    }

    // 4. Intra UV mode against `VP9_KF_UV_MODE_PROB[y_mode]`.
    let uv_probs = &VP9_KF_UV_MODE_PROB[y_mode as usize];
    write_intra_mode(bw, uv_probs, uv_mode);

    // 5. Residual coefs. The decoder's `reconstruct_intra_planes`
    //    walks each plane × each TX position, decoding tokens via
    //    `decode_block_tokens` when `!mi.skip`. We mirror that walk
    //    here. Skip leaves must instead zero the coefficient contexts,
    //    matching libvpx's `dec_reset_skip_context`, because realtime
    //    keyframes can interleave skip and non-skip leaves.
    if !mi.skip {
        write_intra_block_residual(
            bw,
            fc,
            coef_ctx,
            coef_source,
            mi_row,
            mi_col,
            bounds.mi_col_start,
            leaf_bsize,
            // Residual TX walk uses the size actually signalled (==
            // mi.tx_size in the KF select path; the derived cap
            // otherwise), so the token stream matches the decoder's walk.
            effective_tx_size,
            y_mode,
            uv_mode,
            &mi.bmi,
            counts,
        );
    } else {
        reset_skip_coef_context(coef_ctx, leaf_bsize, mi_row, mi_col, bounds.mi_col_start);
    }

    // 6. Splat the just-written neighbour state across the leaf's
    //    mi-coverage. The above/left context for the next block sees
    //    the leaf's Y mode and skip flag.
    let x_mis = usize::from(leaf_bsize.n_8x8_wide());
    let y_mis = usize::from(leaf_bsize.n_8x8_high());
    nbr_ctx.write_block(
        mi_row_in_tile,
        mi_col_in_tile,
        x_mis,
        y_mis,
        y_mode,
        mi.skip,
    );

    Ok(())
}

/// Emit one non-skip keyframe intra leaf's per-plane × per-TX residual
/// stream.
///
/// Mirror of [`write_inter_block_residual`] for the keyframe path —
/// same plane / TX walk, same `write_block_tokens` per TX, just
/// `ref_type = 0` (intra) and the Y-plane `tx_type` is derived from
/// the intra mode (libvpx `intra_mode_to_tx_type_lookup`,
/// `vp9_decodeframe.c:324-325` and the matching path in
/// `rvp9_decoder::reconstruct::resolve_tx_type`). UV always uses
/// `TxType::DctDct`.
///
/// Order matches the decoder's `reconstruct_intra_planes` inner
/// `loff` / `aoff` loop (`rvp9-decoder::tile`, near line 987):
/// for each plane in 0..3, walk plane-block in TX-step-4×4 strides
/// over (row, col).
///
/// The caller must provide a partition tree whose leaves fit the visible
/// frame. Partial superblocks are supported, but partial leaves are not.
#[allow(clippy::too_many_arguments)]
fn write_intra_block_residual(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn IntraCoefSource,
    mi_row: usize,
    mi_col: usize,
    tile_mi_col_start: usize,
    leaf_bsize: BlockSize,
    luma_tx: TxSize,
    y_mode: IntraMode,
    _uv_mode: IntraMode,
    bmi: &[BmiInfo; 4],
    counts: Option<&mut FrameCounts>,
) {
    let (mut coef_bank, mut coef_full_bank) = match counts {
        Some(c) => (Some(&mut c.coef), c.coef_full.as_deref_mut()),
        None => (None, None),
    };
    let mi_col_in_tile = mi_col - tile_mi_col_start;
    let mi_row_in_sb = mi_row & MI_MASK;

    for plane in 0..3 {
        let ss_x = SUBSAMPLING_X[plane];
        let ss_y = SUBSAMPLING_Y[plane];

        let plane_bsize = if plane == 0 {
            leaf_bsize
        } else {
            uv_plane_block_size(leaf_bsize)
        };
        let plane_tx = if plane == 0 {
            luma_tx
        } else {
            uv_plane_tx_size(leaf_bsize, luma_tx)
        };

        // libvpx pd->n4_w / n4_h: sub-8 luma uses `(2 >> ss_x, 2 >>
        // ss_y)` regardless of subsize; ≥ 8×8 uses the plane_bsize
        // 4×4 extent. Mirrors the decoder's
        // `reconstruct_intra_planes` plane-extent derivation.
        let is_sub8 = (leaf_bsize as u8) < (BlockSize::Block8x8 as u8);
        let (plane_bw_4x4, plane_bh_4x4) = if is_sub8 {
            (2 >> ss_x, 2 >> ss_y)
        } else {
            (
                plane_bsize.n_4x4_wide() as usize,
                plane_bsize.n_4x4_high() as usize,
            )
        };
        let tx_step_4x4 = 1usize << (plane_tx as usize);

        // Tile-/SB-relative 4×4 origin in the plane's coef-context
        // tables. Matches the decoder.
        let plane_x4_base = (mi_col_in_tile * 2) >> ss_x;
        let plane_y4_in_sb_base = (mi_row_in_sb * 2) >> ss_y;

        let seg_eob = plane_tx.n_coeffs();

        // Raster-order walk over TX positions inside this plane's
        // coding block.
        let mut tx_y4_local = 0usize;
        while tx_y4_local < plane_bh_4x4 {
            let mut tx_x4_local = 0usize;
            while tx_x4_local < plane_bw_4x4 {
                let mode_for_this_tx = if plane == 0 {
                    keyframe_luma_mode_for_tx(leaf_bsize, bmi, y_mode, tx_x4_local, tx_y4_local)
                } else {
                    y_mode
                };
                let tx_type = if plane == 0 {
                    mode_for_this_tx.tx_type()
                } else {
                    TxType::DctDct
                };
                let (coefs, eob) = coef_source.get_block_coefs(
                    mi_row,
                    mi_col,
                    plane,
                    tx_x4_local,
                    tx_y4_local,
                    plane_tx,
                );
                debug_assert!(
                    coefs.len() >= plane_tx.n_coeffs(),
                    "intra coef source returned slice of len {} < tx.n_coeffs() {} for plane {} tx {:?}",
                    coefs.len(),
                    plane_tx.n_coeffs(),
                    plane,
                    plane_tx,
                );
                debug_assert!(
                    eob <= seg_eob,
                    "intra coef source returned eob {} > seg_eob {} for plane {} tx {:?}",
                    eob,
                    seg_eob,
                    plane,
                    plane_tx,
                );
                write_block_tokens_i16(
                    bw,
                    fc,
                    coef_ctx,
                    plane,
                    plane_x4_base + tx_x4_local,
                    plane_y4_in_sb_base + tx_y4_local,
                    plane_tx,
                    tx_type,
                    /* ref_type = */ 0, // intra
                    seg_eob,
                    coefs,
                    eob,
                    coef_bank.as_deref_mut(),
                    coef_full_bank.as_deref_mut(),
                );
                tx_x4_local += tx_step_4x4;
            }
            tx_y4_local += tx_step_4x4;
        }
    }
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
    /// Tile bounds. VP9 resets spatial mode contexts at every tile
    /// edge, so neighbours outside this rectangle are unavailable even
    /// when the frame-sized grid contains them.
    tile_bounds: TileBounds,
}

impl<'a> MiGridNeighbourLookup<'a> {
    fn new(
        mi_grid: &'a [Option<ModeInfo>],
        mi_cols: usize,
        base_mi_row: usize,
        base_mi_col: usize,
        tile_bounds: TileBounds,
    ) -> Self {
        Self {
            mi_grid,
            mi_cols,
            base_mi_row: base_mi_row as i32,
            base_mi_col: base_mi_col as i32,
            tile_bounds,
        }
    }
}

impl NeighbourLookup for MiGridNeighbourLookup<'_> {
    fn get(&self, pos: Position) -> Option<&ModeInfo> {
        let row = self.base_mi_row.checked_add(i32::from(pos.row))?;
        let col = self.base_mi_col.checked_add(i32::from(pos.col))?;
        if row < self.tile_bounds.mi_row_start as i32
            || col < self.tile_bounds.mi_col_start as i32
            || row >= self.tile_bounds.mi_row_end as i32
            || col >= self.tile_bounds.mi_col_end as i32
        {
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
/// - Every partition leaf must fit inside the visible mi-grid.
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
/// - [`ModesWriterError::UnsupportedIntraInInter`] — (historical;
///   no longer returned after Phase 2b intra-in-inter support).
/// - [`ModesWriterError::InterModeWrite`] — sub-8 leaf, or other
///   inter-mode-writer error.
/// - `InvalidTileBounds` — geometry-level rejection.
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
    seg: Option<SegmentationWriteParams<'_>>,
    prev_mv_grid: Option<&[PrevFrameMvRef]>,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn InterCoefSource,
    counts: Option<&mut FrameCounts>,
) -> Result<(), ModesWriterError> {
    write_tile_inter_with_retained(
        bw,
        fc,
        params,
        tx_mode,
        mi_grid,
        mi_rows,
        mi_cols,
        tile_bounds,
        seg,
        prev_mv_grid,
        coef_ctx,
        coef_source,
        counts,
        None,
    )
}

/// [`write_tile_inter`] with an optional coefficient-token record/replay
/// pass for a two-loop encoder.
#[allow(clippy::too_many_arguments)]
pub fn write_tile_inter_with_retained(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    params: &InterModeReadParams,
    tx_mode: TxMode,
    mi_grid: &[Option<ModeInfo>],
    mi_rows: u32,
    mi_cols: u32,
    tile_bounds: TileBounds,
    seg: Option<SegmentationWriteParams<'_>>,
    prev_mv_grid: Option<&[PrevFrameMvRef]>,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn InterCoefSource,
    mut counts: Option<&mut FrameCounts>,
    mut retained_pass: Option<&mut CoefTokenPass<'_>>,
) -> Result<(), ModesWriterError> {
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
        // SB-row boundary: clear the per-plane left coef-context (the
        // above row persists across SB rows). Mirrors libvpx
        // `vp9_decodeframe.c:2089`.
        coef_ctx.reset_left_coef();

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
                seg,
                prev_mv_grid,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
                retained_pass.as_deref_mut(),
            )?;
            mi_col += MI_BLOCK_SIZE;
        }
        mi_row += MI_BLOCK_SIZE;
    }

    if bw.has_error() {
        return Err(ModesWriterError::BufferOverflow);
    }
    if retained_pass.is_some_and(|pass| !pass.finished()) {
        return Err(ModesWriterError::RetainedCoefMismatch);
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
    seg: Option<SegmentationWriteParams<'_>>,
    prev_mv_grid: Option<&[PrevFrameMvRef]>,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn InterCoefSource,
    mut counts: Option<&mut FrameCounts>,
    mut retained_pass: Option<&mut CoefTokenPass<'_>>,
) -> Result<(), ModesWriterError> {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return Ok(());
    }
    debug_assert!(mi_row < bounds.mi_row_end && mi_col < bounds.mi_col_end);

    let coef_record_only =
        bw.is_count_only() && matches!(retained_pass.as_deref(), Some(CoefTokenPass::Record(_)));

    let n8x8_l2: u8 = n4x4_l2.saturating_sub(1);
    let num_8x8_wh: usize = 1usize << n8x8_l2;
    let hbs: usize = num_8x8_wh >> 1;

    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;

    let has_rows = mi_row + hbs < mi_rows;
    let has_cols = mi_col + hbs < mi_cols;
    let partition = infer_inter_partition(mi_grid, mi_cols, mi_row, mi_col, bsize)?;
    if !coef_record_only {
        let bsl: usize = n8x8_l2 as usize;
        let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);
        set_category(BitCategory::Partition);
        let probs = &fc.partition_prob[ctx];
        if has_rows && has_cols {
            write_partition(bw, probs, partition);
        } else if !has_rows && has_cols {
            debug_assert!(matches!(
                partition,
                PartitionType::Horz | PartitionType::Split
            ));
            bw.write(u8::from(partition == PartitionType::Split), probs[1]);
        } else if has_rows && !has_cols {
            debug_assert!(matches!(
                partition,
                PartitionType::Vert | PartitionType::Split
            ));
            bw.write(u8::from(partition == PartitionType::Split), probs[2]);
        } else {
            debug_assert_eq!(partition, PartitionType::Split);
        }
        // P0b — accumulate partition count for backward adaptation.
        if let Some(c) = counts.as_deref_mut() {
            c.partition[ctx][partition as usize] += 1;
        }
    }

    let sub = subsize(bsize, partition)
        .expect("infer_inter_partition only returns a partition whose subsize is legal");

    // BLOCK_8x8 parent (hbs == 0): non-NONE partitions produce a
    // single sub-8 leaf rather than multiple sub-block leaves.
    // Mirrors the decoder's `decode_inter_partition` `hbs == 0`
    // branch (`tile.rs:1542-1551`).
    if hbs == 0 {
        let mi = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
            .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
        write_inter_block_leaf(
            bw,
            fc,
            params,
            tx_mode,
            mi_grid,
            mi_rows,
            mi_cols,
            *bounds,
            mi_row,
            mi_col,
            sub,
            &mi,
            seg,
            prev_mv_grid,
            coef_ctx,
            coef_source,
            counts.as_deref_mut(),
            retained_pass.as_deref_mut(),
        )?;
        if (bsize as u8) >= (BlockSize::Block8x8 as u8) {
            // Always fires for BLOCK_8x8 — the gate `bsize ==
            // BLOCK_8x8 || partition != SPLIT` is `true` because of
            // the LHS.
            part_ctx.update(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
        }
        return Ok(());
    }

    match partition {
        PartitionType::None => {
            let mi = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_inter_block_leaf(
                bw,
                fc,
                params,
                tx_mode,
                mi_grid,
                mi_rows,
                mi_cols,
                *bounds,
                mi_row,
                mi_col,
                sub,
                &mi,
                seg,
                prev_mv_grid,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
                retained_pass.as_deref_mut(),
            )?;
        }
        PartitionType::Horz => {
            let top = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_inter_block_leaf(
                bw,
                fc,
                params,
                tx_mode,
                mi_grid,
                mi_rows,
                mi_cols,
                *bounds,
                mi_row,
                mi_col,
                sub,
                &top,
                seg,
                prev_mv_grid,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
                retained_pass.as_deref_mut(),
            )?;
            if has_rows {
                let mi_row_b = mi_row + hbs;
                let bot = mi_grid_get(mi_grid, mi_cols, mi_row_b, mi_col).ok_or(
                    ModesWriterError::MissingCell {
                        mi_row: mi_row_b,
                        mi_col,
                    },
                )?;
                write_inter_block_leaf(
                    bw,
                    fc,
                    params,
                    tx_mode,
                    mi_grid,
                    mi_rows,
                    mi_cols,
                    *bounds,
                    mi_row_b,
                    mi_col,
                    sub,
                    &bot,
                    seg,
                    prev_mv_grid,
                    coef_ctx,
                    coef_source,
                    counts.as_deref_mut(),
                    retained_pass.as_deref_mut(),
                )?;
            }
        }
        PartitionType::Vert => {
            let left = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_inter_block_leaf(
                bw,
                fc,
                params,
                tx_mode,
                mi_grid,
                mi_rows,
                mi_cols,
                *bounds,
                mi_row,
                mi_col,
                sub,
                &left,
                seg,
                prev_mv_grid,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
                retained_pass.as_deref_mut(),
            )?;
            if has_cols {
                let mi_col_r = mi_col + hbs;
                let right = mi_grid_get(mi_grid, mi_cols, mi_row, mi_col_r).ok_or(
                    ModesWriterError::MissingCell {
                        mi_row,
                        mi_col: mi_col_r,
                    },
                )?;
                write_inter_block_leaf(
                    bw,
                    fc,
                    params,
                    tx_mode,
                    mi_grid,
                    mi_rows,
                    mi_cols,
                    *bounds,
                    mi_row,
                    mi_col_r,
                    sub,
                    &right,
                    seg,
                    prev_mv_grid,
                    coef_ctx,
                    coef_source,
                    counts.as_deref_mut(),
                    retained_pass.as_deref_mut(),
                )?;
            }
        }
        PartitionType::Split => {
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
                mi_col,
                sub,
                n8x8_l2,
                seg,
                prev_mv_grid,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
                retained_pass.as_deref_mut(),
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
                seg,
                prev_mv_grid,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
                retained_pass.as_deref_mut(),
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
                seg,
                prev_mv_grid,
                coef_ctx,
                coef_source,
                counts.as_deref_mut(),
                retained_pass.as_deref_mut(),
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
                seg,
                prev_mv_grid,
                coef_ctx,
                coef_source,
                counts,
                retained_pass,
            )?;
        }
    }

    // Partition-context update gate — same as the keyframe walker.
    if !coef_record_only
        && (bsize as u8) >= (BlockSize::Block8x8 as u8)
        && (bsize == BlockSize::Block8x8 || partition != PartitionType::Split)
    {
        part_ctx.update(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
    }

    Ok(())
}

#[inline]
fn effective_inter_tx_size(
    mi: &ModeInfo,
    leaf_bsize: BlockSize,
    tx_mode: TxMode,
    is_inter: bool,
) -> TxSize {
    let allow_select = !mi.skip || !is_inter;
    let max_tx = max_txsize(leaf_bsize);
    if allow_select && tx_mode == TxMode::TxModeSelect {
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
    }
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
    tile_bounds: TileBounds,
    mi_row: usize,
    mi_col: usize,
    leaf_bsize: BlockSize,
    mi: &ModeInfo,
    seg: Option<SegmentationWriteParams<'_>>,
    prev_mv_grid: Option<&[PrevFrameMvRef]>,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn InterCoefSource,
    mut counts: Option<&mut FrameCounts>,
    retained_pass: Option<&mut CoefTokenPass<'_>>,
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
    // `seg_id_predicted` is the temporal-update pred-flag bit; AQ.0
    // doesn't emit it (matches libvpx cyclic refresh's
    // `temporal_update = false`). Reject any leaf that has it set.
    if mi.seg_id_predicted {
        return Err(ModesWriterError::UnsupportedSegmentation { mi_row, mi_col });
    }
    // When the writer scope is "no segmentation", reject any non-zero
    // segment_id — the wire format can't carry it.
    if seg.is_none() && mi.segment_id != 0 {
        return Err(ModesWriterError::UnsupportedSegmentation { mi_row, mi_col });
    }
    let is_inter = matches!(mi.mode, BlockMode::Inter { .. });
    let effective_tx_size = effective_inter_tx_size(mi, leaf_bsize, tx_mode, is_inter);

    let coef_record_only =
        bw.is_count_only() && matches!(retained_pass.as_deref(), Some(CoefTokenPass::Record(_)));
    if coef_record_only {
        write_or_reset_inter_residual(
            bw,
            fc,
            coef_ctx,
            coef_source,
            mi_row,
            mi_col,
            tile_bounds.mi_col_start,
            leaf_bsize,
            effective_tx_size,
            mi,
            counts,
            retained_pass,
        );
        return Ok(());
    }

    // 1. Segment_id. AQ.0 emits a plain tree decode when
    //    `seg.is_some()` (i.e. `enabled && update_map &&
    //    !temporal_update`). The decoder mirror is
    //    `read_inter_segment_id` at the `update_map && !temporal_update`
    //    branch (`mode_info.rs:160-165`).
    if let Some(seg) = seg {
        set_category(BitCategory::SegId);
        write_segment_id(bw, mi.segment_id, seg.tree_probs);
    }

    // Build the neighbour lookup adapter for this leaf. The decoder
    // builds the equivalent `TileNeighbourLookup` rooted at the
    // current block's top-left.
    let neighbours = MiGridNeighbourLookup::new(mi_grid, mi_cols, mi_row, mi_col, tile_bounds);
    let above = neighbours.get(Position { row: -1, col: 0 });
    let left = neighbours.get(Position { row: 0, col: -1 });

    // 2. Skip flag. Pre-S4.16g this was hardcoded to `1`; phase 3
    //    of S4.16g widened the writer to accept `mi.skip == false`.
    //    Mirrors `decode_inter_block` step 2.
    let skip_ctx =
        usize::from(above.is_some_and(|m| m.skip)) + usize::from(left.is_some_and(|m| m.skip));
    set_category(BitCategory::Skip);
    bw.write(u8::from(mi.skip), fc.skip_probs[skip_ctx]);
    // P0b — accumulate skip count.
    if let Some(c) = counts.as_deref_mut() {
        c.skip[skip_ctx][usize::from(mi.skip)] += 1;
    }

    // 3. is_inter classification. Phase 2b: 0 for intra-in-inter
    //    blocks, 1 for inter blocks.
    let intra_inter_ctx = get_intra_inter_context(above, left);
    set_category(BitCategory::IsInter);
    bw.write(u8::from(is_inter), fc.intra_inter_prob[intra_inter_ctx]);
    // P0b — accumulate intra_inter count.
    if let Some(c) = counts.as_deref_mut() {
        c.intra_inter[intra_inter_ctx][usize::from(is_inter)] += 1;
    }

    // 4. tx_size. `allow_select = !skip || !is_inter` per libvpx
    //    `vp9_decodemv.c:800`. For intra blocks `!is_inter = true`
    //    so `allow_select = true` always. For inter blocks with
    //    `skip = true`, `allow_select = false`.
    let allow_select = !mi.skip || !is_inter;
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
    set_category(BitCategory::TxSize);
    crate::mode_info::write_tx_size(
        bw,
        &fc.tx_probs,
        tx_mode,
        max_tx,
        allow_select,
        tx_ctx,
        effective_tx_size,
    );
    // P0b — accumulate tx-size count. Mirrors the decoder's
    // `read_tx_size` count logic in `mode_info.rs`: the per-context
    // p-counts (which `adapt_mode_probs` consumes for tx_probs backward
    // adaptation) are incremented ONLY in the selected-tx-size branch
    // (`allow_select && tx_mode == TxModeSelect && max_tx >= Tx8x8`).
    // In the non-select branch libvpx (and the decoder) skip the
    // increment, so the writer must too — otherwise skip-inter leaves
    // (allow_select == false) over-count vs the decoder and desync the
    // adapted tx_probs across the GOP. `tx_totals` is encoder-only
    // bookkeeping (the decoder never tracks it); keep it accumulating
    // for parity with libvpx's tokenize accounting.
    let tx_counts_selected = allow_select && tx_mode == TxMode::TxModeSelect;
    if let Some(c) = counts.as_deref_mut() {
        let tx_idx = effective_tx_size as usize;
        match max_tx {
            TxSize::Tx32x32 => {
                if tx_counts_selected {
                    c.tx.p32x32[tx_ctx][tx_idx] += 1;
                }
                c.tx.tx_totals[tx_idx] += 1;
            }
            TxSize::Tx16x16 => {
                if tx_counts_selected {
                    c.tx.p16x16[tx_ctx][tx_idx] += 1;
                }
                c.tx.tx_totals[tx_idx] += 1;
            }
            TxSize::Tx8x8 => {
                if tx_counts_selected {
                    c.tx.p8x8[tx_ctx][tx_idx] += 1;
                }
                c.tx.tx_totals[tx_idx] += 1;
            }
            TxSize::Tx4x4 => {} // no choice to count
        }
    }

    // 5. Dispatch: intra-in-inter vs inter mode info.
    if is_inter {
        // 5a. Inter-block mode info. Build the per-leaf `BlockBounds`
        //     the `find_mv_refs` MV clamp inside
        //     `write_inter_block_mode_info` needs (libvpx
        //     `set_mi_row_col`).
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
        set_category(BitCategory::ModeRef);
        write_inter_block_mode_info(
            bw,
            fc,
            params,
            &neighbours,
            leaf_bsize,
            &bounds_block,
            prev_mv,
            mi,
            counts.as_deref_mut(),
        )
        .map_err(|e| ModesWriterError::InterModeWrite {
            mi_row,
            mi_col,
            source: e,
        })?;
    } else {
        // 5b. Phase 2b — intra-in-inter mode info. Mirrors the
        //     decoder's `read_intra_in_inter_mode_info`
        //     (`tile.rs:2457`). For >= 8x8 blocks: write a single
        //     y_mode against `fc.y_mode_prob[size_group]`, then
        //     uv_mode against `fc.uv_mode_prob[y_mode]`.
        let (y_mode, uv_mode) = match mi.mode {
            BlockMode::Intra { y_mode, uv_mode } => (y_mode, uv_mode),
            _ => unreachable!("is_inter == false but mode is Inter"),
        };

        // size_group_lookup — mirrors libvpx `vp9_common_data.c:32-33`
        // and the decoder's inline mapping in
        // `read_intra_in_inter_mode_info`.
        let size_group = match leaf_bsize {
            BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4 => 0,
            BlockSize::Block8x8 | BlockSize::Block8x16 | BlockSize::Block16x8 => 1,
            BlockSize::Block16x16 | BlockSize::Block16x32 | BlockSize::Block32x16 => 2,
            BlockSize::Block32x32
            | BlockSize::Block32x64
            | BlockSize::Block64x32
            | BlockSize::Block64x64 => 3,
        };

        set_category(BitCategory::ModeRef);

        // Y mode — tree-coded against `fc.y_mode_prob[size_group]`.
        write_intra_mode(bw, &fc.y_mode_prob[size_group], y_mode);
        // P0b — accumulate y_mode count.
        if let Some(c) = counts.as_deref_mut() {
            c.y_mode[size_group][y_mode as usize] += 1;
        }

        // UV mode — tree-coded against `fc.uv_mode_prob[y_mode]`.
        write_intra_mode(bw, &fc.uv_mode_prob[y_mode as usize], uv_mode);
        // P0b — accumulate uv_mode count.
        if let Some(c) = counts.as_deref_mut() {
            c.uv_mode[y_mode as usize][uv_mode as usize] += 1;
        }
    }

    // 6. Residual coefs. When `mi.skip == true` the decoder bypasses
    //    the per-TX residual decode entirely; we emit nothing. When
    //    `mi.skip == false` we walk planes 0/1/2 and emit one
    //    `write_block_tokens` per TX position, mirroring the decoder's
    //    `reconstruct_inter_block` / `reconstruct_intra_planes` loop.
    write_or_reset_inter_residual(
        bw,
        fc,
        coef_ctx,
        coef_source,
        mi_row,
        mi_col,
        tile_bounds.mi_col_start,
        leaf_bsize,
        effective_tx_size,
        mi,
        counts,
        retained_pass,
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_or_reset_inter_residual(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn InterCoefSource,
    mi_row: usize,
    mi_col: usize,
    tile_mi_col_start: usize,
    leaf_bsize: BlockSize,
    effective_tx_size: TxSize,
    mi: &ModeInfo,
    counts: Option<&mut FrameCounts>,
    retained_pass: Option<&mut CoefTokenPass<'_>>,
) {
    if !mi.skip {
        let is_inter = matches!(mi.mode, BlockMode::Inter { .. });
        // G5 -- extract y_mode for intra-in-inter residual coding.
        // Inter blocks pass IntraMode::Dc (unused since is_inter
        // forces DctDct + ref_type=1 inside the residual writer).
        let residual_y_mode = match mi.mode {
            BlockMode::Intra { y_mode, .. } => y_mode,
            _ => IntraMode::Dc,
        };
        set_category(BitCategory::Coef);
        // A6 TWO_LOOP — split the `FrameCounts` borrow into the 4-token
        // `.coef` model bank (backward adaptation) and the optional
        // encoder-only full 12-token `.coef_full` bank (count pass).
        let (coef_bank, coef_full_bank): (
            Option<&mut CoefCountsBank>,
            Option<&mut crate::entropy::adapt::Coef12Bank>,
        ) = match counts {
            Some(c) => (Some(&mut c.coef), c.coef_full.as_deref_mut()),
            None => (None, None),
        };
        write_inter_block_residual(
            bw,
            fc,
            coef_ctx,
            coef_source,
            mi_row,
            mi_col,
            tile_mi_col_start,
            leaf_bsize,
            effective_tx_size,
            is_inter,
            residual_y_mode,
            coef_bank,
            coef_full_bank,
            retained_pass,
        );
    } else {
        // RC.X1 — mirror libvpx's `dec_reset_skip_context` so the next
        // non-skip leaf derives the same initial coefficient context.
        reset_skip_coef_context(coef_ctx, leaf_bsize, mi_row, mi_col, tile_mi_col_start);
    }
}

/// Record one fully encoded inter-frame leaf's coefficient tokens.
///
/// This is the coefficient-only portion of [`write_inter_block_leaf`]. It
/// lets a realtime encoder tokenize a leaf immediately after transform and
/// quantization, while the owning tile job is already active. The caller must
/// invoke leaves in the same per-tile syntax order as [`write_tile_inter`].
/// Skip leaves update coefficient contexts without recording EOB symbols,
/// matching the decoder's `dec_reset_skip_context` path.
#[allow(clippy::too_many_arguments)]
pub fn record_inter_leaf_coefficients(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    tx_mode: TxMode,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn InterCoefSource,
    mi_row: usize,
    mi_col: usize,
    tile_mi_col_start: usize,
    leaf_bsize: BlockSize,
    mi: &ModeInfo,
    counts: &mut FrameCounts,
    retained: &mut RetainedCoefTokens,
) {
    debug_assert_eq!(mi.block_size, leaf_bsize);
    let is_inter = matches!(mi.mode, BlockMode::Inter { .. });
    let effective_tx_size = effective_inter_tx_size(mi, leaf_bsize, tx_mode, is_inter);
    if mi.skip {
        reset_skip_coef_context(coef_ctx, leaf_bsize, mi_row, mi_col, tile_mi_col_start);
        return;
    }

    let y_mode = match mi.mode {
        BlockMode::Intra { y_mode, .. } => y_mode,
        _ => IntraMode::Dc,
    };
    debug_assert!(bw.is_count_only());
    let mut token_pass = CoefTokenPass::record(retained);
    let coef_full = counts.coef_full.as_deref_mut();
    write_inter_block_residual(
        bw,
        fc,
        coef_ctx,
        coef_source,
        mi_row,
        mi_col,
        tile_mi_col_start,
        leaf_bsize,
        effective_tx_size,
        is_inter,
        y_mode,
        Some(&mut counts.coef),
        coef_full,
        Some(&mut token_pass),
    );
}

/// RC.X1 — encoder-side mirror of libvpx's `dec_reset_skip_context`
/// (`vp9_decodeframe.c:1008-1010`). When a leaf is skip-coded the
/// decoder zeros the per-plane above/left coef-context cells the
/// block covers; we must do the same so subsequent non-skip leaves
/// initialize their `initial_pt` from the same (zeroed) neighbour
/// context the decoder sees.
fn reset_skip_coef_context(
    coef_ctx: &mut TileCoefCtx,
    leaf_bsize: BlockSize,
    mi_row: usize,
    mi_col: usize,
    tile_mi_col_start: usize,
) {
    let mi_col_in_tile = mi_col - tile_mi_col_start;
    let mi_row_in_sb = mi_row & MI_MASK;
    for plane in 0..3 {
        let ss_x = SUBSAMPLING_X[plane];
        let ss_y = SUBSAMPLING_Y[plane];

        let plane_bsize = if plane == 0 {
            leaf_bsize
        } else {
            uv_plane_block_size(leaf_bsize)
        };

        // Mirrors `write_inter_block_residual`'s plane-extent
        // derivation. Sub-8 luma leaves zero a 2x2 4×4 cell tile (the
        // parent 8×8's coverage); ≥8×8 leaves zero the plane_bsize
        // 4×4-cell extent.
        let is_sub8 = (leaf_bsize as u8) < (BlockSize::Block8x8 as u8);
        let (plane_bw_4x4, plane_bh_4x4) = if is_sub8 {
            (2 >> ss_x, 2 >> ss_y)
        } else {
            (
                plane_bsize.n_4x4_wide() as usize,
                plane_bsize.n_4x4_high() as usize,
            )
        };

        let plane_x4_base = (mi_col_in_tile * 2) >> ss_x;
        let plane_y4_in_sb_base = (mi_row_in_sb * 2) >> ss_y;

        // Zero (above × left) cells covering the block. `set_coef_flags`
        // with `nonzero = false` writes 0 to every covered cell —
        // exactly what `memset(..., 0, ...)` does in libvpx.
        coef_ctx.set_coef_flags(
            plane,
            plane_x4_base,
            plane_y4_in_sb_base,
            plane_bw_4x4,
            plane_bh_4x4,
            /* nonzero = */ false,
        );
    }
}

/// Per-plane chroma subsampling for VP9 Profile 0 (4:2:0). Plane 0
/// (Y) is full-resolution; planes 1/2 (U/V) are half-resolution in
/// both axes. Mirrors `rvp9_decoder::reconstruct_inter::SUBSAMPLING_*`.
const SUBSAMPLING_X: [usize; 3] = [0, 1, 1];
const SUBSAMPLING_Y: [usize; 3] = [0, 1, 1];

/// libvpx `ss_size_lookup[bsize][1][1]` — chroma plane's block size
/// for a luma `bsize` under 4:2:0. Mirrors
/// `rvp9_decoder::reconstruct_inter::uv_plane_block_size`.
fn uv_plane_block_size(bsize: BlockSize) -> BlockSize {
    use BlockSize::*;
    match bsize {
        Block4x4 | Block4x8 | Block8x4 | Block8x8 => Block4x4,
        Block8x16 => Block4x8,
        Block16x8 => Block8x4,
        Block16x16 => Block8x8,
        Block16x32 => Block8x16,
        Block32x16 => Block16x8,
        Block32x32 => Block16x16,
        Block32x64 => Block16x32,
        Block64x32 => Block32x16,
        Block64x64 => Block32x32,
    }
}

/// libvpx `get_uv_tx_size`: `min(luma_tx, max_txsize_lookup[uv_bsize])`.
/// Mirrors `rvp9_decoder::reconstruct_inter::uv_plane_tx_size`.
fn uv_plane_tx_size(bsize: BlockSize, luma_tx: TxSize) -> TxSize {
    let cap = max_txsize(uv_plane_block_size(bsize));
    if (luma_tx as u8) < (cap as u8) {
        luma_tx
    } else {
        cap
    }
}

/// Emit one non-skip inter leaf's per-plane × per-TX residual stream.
///
/// Mirrors the decoder's `reconstruct_inter_block` residual loop
/// (`rvp9-decoder::reconstruct_inter`, lines 505-543):
///
/// For each plane in `0..3`:
///
/// - Derive `plane_bsize` (`bsize` for Y, `uv_plane_block_size` for UV).
/// - Derive per-plane `plane_tx` (`mi.tx_size` for Y, capped via
///   `uv_plane_tx_size` for UV).
/// - Step in `tx_step_4x4 = 1 << plane_tx` 4×4-sub-block units across the
///   plane block.
/// - For each TX position emit one `write_block_tokens` against
///   `TxType::DctDct` and `ref_type = 1` (inter).
///
/// The `tx_x4_local` / `tx_y4_local` coordinates passed to the
/// `InterCoefSource` are plane-leaf-relative (so `(0, 0)` is the
/// leaf's top-left in this plane). The tile-absolute coordinates the
/// `TileCoefCtx` consumes are derived inside, matching the decoder's
/// `plane_x4_base = (mi_col_in_tile * 2) >> ss_x` /
/// `plane_y4_in_sb_base = (mi_row_in_sb * 2) >> ss_y`.
///
/// `tile_mi_col_start` converts the frame-relative block column to the
/// tile-relative coefficient-context column. `mi_row_in_sb` remains
/// `mi_row & MI_MASK` because tile rows are superblock-aligned.
///
/// The caller must provide a partition tree whose leaves fit the visible
/// frame. Partial superblocks are supported, but partial leaves are not.
#[allow(clippy::too_many_arguments)]
fn write_inter_block_residual(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    coef_ctx: &mut TileCoefCtx,
    coef_source: &dyn InterCoefSource,
    mi_row: usize,
    mi_col: usize,
    tile_mi_col_start: usize,
    leaf_bsize: BlockSize,
    luma_tx: TxSize,
    // G5 -- intra-in-inter blocks use non-DctDct tx_type and intra
    // probability tables (ref_type = 0). `is_inter` and `y_mode`
    // let the per-TX loop derive the correct values.
    is_inter: bool,
    y_mode: IntraMode,
    // G3 -- coef count accumulation for backward adaptation.
    mut counts: Option<&mut CoefCountsBank>,
    // A6 TWO_LOOP -- encoder-only full 12-token histogram (count pass).
    mut full_counts: Option<&mut crate::entropy::adapt::Coef12Bank>,
    mut retained_pass: Option<&mut CoefTokenPass<'_>>,
) {
    let mi_col_in_tile = mi_col - tile_mi_col_start;
    let mi_row_in_sb = mi_row & MI_MASK;
    let ref_type = usize::from(is_inter);

    for plane in 0..3 {
        let ss_x = SUBSAMPLING_X[plane];
        let ss_y = SUBSAMPLING_Y[plane];

        let plane_bsize = if plane == 0 {
            leaf_bsize
        } else {
            uv_plane_block_size(leaf_bsize)
        };
        let plane_tx = if plane == 0 {
            luma_tx
        } else {
            uv_plane_tx_size(leaf_bsize, luma_tx)
        };

        // G5 -- tx_type: inter blocks always DctDct. Intra-in-inter
        // Y plane (non-32x32) uses the intra mode's hybrid type.
        // UV planes and Tx32x32 always DctDct. Matches the decoder's
        // `resolve_tx_type`.
        let tx_type = if is_inter {
            TxType::DctDct
        } else if plane == 0 && plane_tx != TxSize::Tx32x32 {
            y_mode.tx_type()
        } else {
            TxType::DctDct
        };

        // For sub-8 luma the decoder pins the plane extent to
        // `(2 >> ss_x, 2 >> ss_y)` -- the parent 8x8's 4x4 cell count.
        // For >=8x8 we use the plane_bsize directly. Mirrors
        // `reconstruct_inter_block`'s `(plane_bw_4x4, plane_bh_4x4)`
        // derivation (line 218).
        let is_sub8 = (leaf_bsize as u8) < (BlockSize::Block8x8 as u8);
        let (plane_bw_4x4, plane_bh_4x4) = if is_sub8 {
            (2 >> ss_x, 2 >> ss_y)
        } else {
            (
                plane_bsize.n_4x4_wide() as usize,
                plane_bsize.n_4x4_high() as usize,
            )
        };
        let tx_step_4x4 = 1usize << (plane_tx as usize);

        // Tile- / SB-relative 4x4 origin in the plane's coef-context
        // tables. Matches the decoder's `plane_x4_base` /
        // `plane_y4_in_sb_base` formulas.
        let plane_x4_base = (mi_col_in_tile * 2) >> ss_x;
        let plane_y4_in_sb_base = (mi_row_in_sb * 2) >> ss_y;

        let seg_eob = plane_tx.n_coeffs();

        // Raster-order walk over TX positions inside this plane's
        // coding block.
        let mut tx_y4_local = 0usize;
        while tx_y4_local < plane_bh_4x4 {
            let mut tx_x4_local = 0usize;
            while tx_x4_local < plane_bw_4x4 {
                let (coefs, eob) = coef_source.get_block_coefs(
                    mi_row,
                    mi_col,
                    plane,
                    tx_x4_local,
                    tx_y4_local,
                    plane_tx,
                );
                debug_assert!(
                    coefs.len() >= plane_tx.n_coeffs(),
                    "coef source returned slice of len {} < tx.n_coeffs() {} for plane {} tx {:?}",
                    coefs.len(),
                    plane_tx.n_coeffs(),
                    plane,
                    plane_tx,
                );
                debug_assert!(
                    eob <= seg_eob,
                    "coef source returned eob {} > seg_eob {} for plane {} tx {:?}",
                    eob,
                    seg_eob,
                    plane,
                    plane_tx,
                );
                write_block_tokens_with_retained_i16(
                    bw,
                    fc,
                    coef_ctx,
                    plane,
                    plane_x4_base + tx_x4_local,
                    plane_y4_in_sb_base + tx_y4_local,
                    plane_tx,
                    tx_type,
                    ref_type,
                    seg_eob,
                    coefs,
                    eob,
                    counts.as_deref_mut(),
                    full_counts.as_deref_mut(),
                    retained_pass.as_deref_mut(),
                );
                tx_x4_local += tx_step_4x4;
            }
            tx_y4_local += tx_step_4x4;
        }
    }
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

    /// Test-only wrapper around the widened `write_tile_intra_keyframe`
    /// signature. Supplies a stack-local `TileCoefCtx` + an
    /// `AllZeroIntraCoefSource` so the existing tests (which all assume
    /// `skip = true` and don't care about residual emission) don't have
    /// to thread those parameters individually.
    fn write_tile_intra_keyframe_default(
        bw: &mut BoolWriter<'_>,
        mi_grid: &[Option<ModeInfo>],
        mi_rows: u32,
        mi_cols: u32,
        tile_bounds: TileBounds,
        fc: &FrameContext,
    ) -> Result<(), ModesWriterError> {
        let tile_mi_cols = tile_bounds
            .mi_col_end
            .saturating_sub(tile_bounds.mi_col_start)
            .max(1);
        let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        write_tile_intra_keyframe(
            bw,
            mi_grid,
            mi_rows,
            mi_cols,
            tile_bounds,
            fc,
            // Existing tests pin cells to Tx32x32 + the AllowTx32x32 non-
            // select mode, so no tx_size bits are emitted (the decoder
            // derives the size). The Track-F iter8 select-path bytes are
            // exercised by the cross-decoder gate + drift_diag harnesses.
            TxMode::AllowTx32x32,
            /* seg = */ None,
            &mut coef_ctx,
            &coef_source,
        )
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

    #[test]
    fn tile_neighbour_lookup_hides_frame_neighbour_across_column_boundary() {
        let grid = make_dc_grid(8, 16);
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 8,
            mi_col_end: 16,
        };
        let lookup = MiGridNeighbourLookup::new(&grid, 16, 1, 8, bounds);

        assert!(
            lookup.get(Position { row: 0, col: -1 }).is_none(),
            "the previous frame column belongs to another tile"
        );
        assert!(
            lookup.get(Position { row: -1, col: 0 }).is_some(),
            "the above cell remains available inside this tile"
        );
    }

    #[test]
    fn tile_bounds_follow_superblock_aligned_libvpx_geometry() {
        let two_cols_320 = [
            tile_bounds(30, 40, 0, 1, 0, 0),
            tile_bounds(30, 40, 0, 1, 0, 1),
        ];
        assert_eq!(
            two_cols_320.map(|b| (b.mi_col_start, b.mi_col_end)),
            [(0, 16), (16, 40)]
        );

        let two_cols_640 = [
            tile_bounds(45, 80, 0, 1, 0, 0),
            tile_bounds(45, 80, 0, 1, 0, 1),
        ];
        assert_eq!(
            two_cols_640.map(|b| (b.mi_col_start, b.mi_col_end)),
            [(0, 40), (40, 80)]
        );

        let four_cols_1280 =
            core::array::from_fn::<_, 4, _>(|col| tile_bounds(88, 160, 0, 2, 0, col as u32));
        assert_eq!(
            four_cols_1280.map(|b| (b.mi_col_start, b.mi_col_end)),
            [(0, 40), (40, 80), (80, 120), (120, 160)]
        );

        let four_cols_1920 =
            core::array::from_fn::<_, 4, _>(|col| tile_bounds(135, 240, 0, 2, 0, col as u32));
        assert_eq!(
            four_cols_1920.map(|b| (b.mi_col_start, b.mi_col_end)),
            [(0, 56), (56, 120), (120, 176), (176, 240)]
        );
    }

    #[test]
    fn inter_residual_uses_tile_local_coefficient_columns() {
        let mut buf = vec![0u8; 16 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let fc = FrameContext::new_default();
        let mut coef_ctx = TileCoefCtx::new(8);
        coef_ctx.set_coef_flags(0, 0, 0, 8, 8, true);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();

        write_inter_block_residual(
            &mut bw,
            &fc,
            &mut coef_ctx,
            &coef_source,
            0,
            8,
            8,
            BlockSize::Block64x64,
            TxSize::Tx32x32,
            true,
            IntraMode::Dc,
            None,
            None,
            None,
        );

        assert!(
            (0..8).all(|x4| coef_ctx.above_coef(0, x4) == 0),
            "the second tile's first block must update local columns 0..8"
        );
    }

    #[test]
    fn keyframe_residual_uses_tile_local_coefficient_columns() {
        let mut buf = vec![0u8; 16 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let fc = FrameContext::new_default();
        let mut coef_ctx = TileCoefCtx::new(8);
        coef_ctx.set_coef_flags(0, 0, 0, 8, 8, true);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();

        write_intra_block_residual(
            &mut bw,
            &fc,
            &mut coef_ctx,
            &coef_source,
            0,
            8,
            8,
            BlockSize::Block64x64,
            TxSize::Tx32x32,
            IntraMode::Dc,
            IntraMode::Dc,
            &[BmiInfo::default(); 4],
            None,
        );

        assert!(
            (0..8).all(|x4| coef_ctx.above_coef(0, x4) == 0),
            "the second tile's first block must update local columns 0..8"
        );
    }

    #[test]
    fn keyframe_skip_resets_luma_and_chroma_coefficient_contexts() {
        let mut coef_ctx = TileCoefCtx::new(8);
        for (plane, &subsampling_x) in SUBSAMPLING_X.iter().enumerate() {
            coef_ctx.set_coef_flags(plane, 0, 0, 16 >> subsampling_x, 8, true);
        }

        reset_skip_coef_context(&mut coef_ctx, BlockSize::Block8x8, 1, 2, 0);

        assert_eq!(coef_ctx.above_coef(0, 4), 0);
        assert_eq!(coef_ctx.above_coef(0, 5), 0);
        assert_eq!(coef_ctx.left_coef(0, 2), 0);
        assert_eq!(coef_ctx.left_coef(0, 3), 0);
        assert_eq!(coef_ctx.above_coef(1, 2), 0);
        assert_eq!(coef_ctx.left_coef(1, 1), 0);
        assert_eq!(coef_ctx.above_coef(2, 2), 0);
        assert_eq!(coef_ctx.left_coef(2, 1), 0);
        assert_eq!(coef_ctx.above_coef(0, 3), 1);
        assert_eq!(coef_ctx.left_coef(0, 1), 1);
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
            let mut coef_ctx = TileCoefCtx::new(8);
            let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
            write_block_intra_keyframe(
                &mut bw,
                &fc,
                TxMode::AllowTx32x32,
                /* seg = */ None,
                &mut nbr_ctx,
                &bounds,
                &grid,
                8,
                8,
                0,
                0,
                BlockSize::Block64x64,
                &mi,
                &mut coef_ctx,
                &coef_source,
                None,
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
        let err = write_tile_intra_keyframe_default(&mut bw, &grid, 8, 8, bounds, &fc).unwrap_err();
        assert!(matches!(
            err,
            ModesWriterError::MissingCell {
                mi_row: 0,
                mi_col: 0
            }
        ));
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
        let err = write_tile_intra_keyframe_default(&mut bw, &grid, 8, 8, bounds, &fc).unwrap_err();
        assert!(
            matches!(err, ModesWriterError::InconsistentPartitionTree { .. })
                || matches!(err, ModesWriterError::UnsupportedBlockSize { .. }),
            "got: {err:?}"
        );
    }

    /// `write_tile_intra_keyframe` rejects a non-skip cell.
    /// F5.1 — the keyframe writer accepts `skip = false` cells and
    /// emits residual coefs via the `IntraCoefSource` plumbing.
    /// Companion to `s4_16k_round_trip_non_skip_all_zero_through_decoder`
    /// in the rvp9-decoder crate (which exercises the same path for the
    /// inter writer). With `AllZeroIntraCoefSource` every TX block
    /// reports `eob = 0`, so the writer emits a single terminal-EOB
    /// bit per (plane × TX) — 6 EOB bits per 64×64 SB (4 Y Tx32x32 +
    /// 1 U + 1 V) on top of the mode-info bits.
    #[test]
    fn write_tile_intra_keyframe_accepts_non_skip_with_all_zero_coefs() {
        let mut buf = vec![0u8; 1024];
        let fc = FrameContext::new_default();
        let mut grid = make_dc_grid(8, 8);
        if let Some(c) = grid[0].as_mut() {
            c.skip = false;
        }
        // mi-grid splat: every 8×8 cell of the 64×64 SB carries the
        // same ModeInfo. (The walker reads `block_size` from the
        // top-left cell to infer PARTITION_NONE; the rest are tracked
        // for the leaf's coverage.)
        for cell in grid.iter_mut().skip(1) {
            if let Some(c) = cell.as_mut() {
                c.skip = false;
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(8);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        write_tile_intra_keyframe(
            &mut bw,
            &grid,
            8,
            8,
            bounds,
            &fc,
            TxMode::AllowTx32x32,
            /* seg = */ None,
            &mut coef_ctx,
            &coef_source,
        )
        .expect("non-skip + all-zero coef source must succeed");
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
        let err = write_tile_intra_keyframe_default(&mut bw, &grid, 8, 8, bounds, &fc).unwrap_err();
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

    /// A seven-mi-row keyframe uses the reduced bottom-edge partition
    /// syntax and round-trips without visiting an out-of-frame leaf.
    #[test]
    fn write_tile_intra_keyframe_round_trips_partial_bottom_sb() {
        let mi_rows = 7u32;
        let mi_cols = 8u32;
        let mut grid = vec![None; (mi_rows * mi_cols) as usize];
        for col in [0, 4] {
            splat_cell(
                &mut grid,
                mi_cols as usize,
                0,
                col,
                make_dc_cell(BlockSize::Block32x32),
            );
        }
        for col in [0, 2, 4, 6] {
            splat_cell(
                &mut grid,
                mi_cols as usize,
                4,
                col,
                make_dc_cell(BlockSize::Block16x16),
            );
        }
        for col in 0..8 {
            splat_cell(
                &mut grid,
                mi_cols as usize,
                6,
                col,
                make_dc_cell(BlockSize::Block8x8),
            );
        }

        let leaves = encode_then_replay(mi_rows, mi_cols, &grid);
        assert_eq!(leaves.len(), 14);
        assert!(
            leaves.iter().all(|(row, _, bsize)| {
                row + usize::from(bsize.n_8x8_high()) <= mi_rows as usize
            })
        );
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
            write_tile_intra_keyframe_default(
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
        let probs_for_above_left =
            |above: IntraMode, left: IntraMode| VP9_KF_Y_MODE_PROB[above as usize][left as usize];

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
                    write_tile_intra_keyframe_default(
                        &mut bw, &grid, mi_rows, mi_cols, bounds, &fc,
                    )
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
                assert_eq!(
                    got_uv, uv_mode,
                    "UV mode parse for ({y_mode:?}, {uv_mode:?})"
                );
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
        if mi_row >= bounds.mi_row_end || mi_col >= bounds.mi_col_end {
            return;
        }
        let n8x8_l2 = n4x4_l2.saturating_sub(1);
        let num_8x8_wh: usize = 1usize << n8x8_l2;
        let hbs = num_8x8_wh >> 1;
        let has_rows = mi_row + hbs < bounds.mi_row_end;
        let has_cols = mi_col + hbs < bounds.mi_col_end;
        let mi_row_in_tile = mi_row - bounds.mi_row_start;
        let mi_col_in_tile = mi_col - bounds.mi_col_start;
        let bsl = n8x8_l2 as usize;
        let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);
        let probs = &VP9_KF_PARTITION_PROBS[ctx];
        let partition = if has_rows && has_cols {
            read_partition(r, probs, ctx, None)
        } else if !has_rows && has_cols {
            if r.read(probs[1]) != 0 {
                PartitionType::Split
            } else {
                PartitionType::Horz
            }
        } else if has_rows && !has_cols {
            if r.read(probs[2]) != 0 {
                PartitionType::Split
            } else {
                PartitionType::Vert
            }
        } else {
            PartitionType::Split
        };
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
                if has_rows {
                    read_leaf(r, nbr, mi_row + hbs, mi_col, sub);
                }
            }
            PartitionType::Vert => {
                read_leaf(r, nbr, mi_row, mi_col, sub);
                if has_cols {
                    read_leaf(r, nbr, mi_row, mi_col + hbs, sub);
                }
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
            write_tile_intra_keyframe_default(&mut bw, grid, mi_rows, mi_cols, bounds, &fc)
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
        splat_cell(
            &mut grid,
            mi_cols as usize,
            0,
            0,
            make_dc_cell(BlockSize::Block64x32),
        );
        splat_cell(
            &mut grid,
            mi_cols as usize,
            4,
            0,
            make_dc_cell(BlockSize::Block64x32),
        );

        let leaves = encode_then_replay(mi_rows, mi_cols, &grid);
        assert_eq!(
            leaves,
            vec![(0, 0, BlockSize::Block64x32), (4, 0, BlockSize::Block64x32),]
        );
    }

    /// 64x64 PARTITION_VERT — two `Block32x64` leaves side-by-side.
    #[test]
    fn partition_vert_64x64_round_trip() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        splat_cell(
            &mut grid,
            mi_cols as usize,
            0,
            0,
            make_dc_cell(BlockSize::Block32x64),
        );
        splat_cell(
            &mut grid,
            mi_cols as usize,
            0,
            4,
            make_dc_cell(BlockSize::Block32x64),
        );

        let leaves = encode_then_replay(mi_rows, mi_cols, &grid);
        assert_eq!(
            leaves,
            vec![(0, 0, BlockSize::Block32x64), (0, 4, BlockSize::Block32x64),]
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
        splat_cell(
            &mut grid,
            mi_cols,
            0,
            0,
            make_dc_cell(BlockSize::Block64x64),
        );
        let p = infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block64x64).expect("legal NONE");
        assert_eq!(p, PartitionType::None);
    }

    /// `infer_partition` returns `Horz` when the leaf at the top-left
    /// matches `subsize(bsize, HORZ)`.
    #[test]
    fn infer_partition_horz() {
        let mi_cols = 8usize;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; mi_cols * mi_cols];
        splat_cell(
            &mut grid,
            mi_cols,
            0,
            0,
            make_dc_cell(BlockSize::Block64x32),
        );
        let p = infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block64x64).expect("legal HORZ");
        assert_eq!(p, PartitionType::Horz);
    }

    /// `infer_partition` returns `Vert` when the leaf at the top-left
    /// matches `subsize(bsize, VERT)`.
    #[test]
    fn infer_partition_vert() {
        let mi_cols = 8usize;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; mi_cols * mi_cols];
        splat_cell(
            &mut grid,
            mi_cols,
            0,
            0,
            make_dc_cell(BlockSize::Block32x64),
        );
        let p = infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block64x64).expect("legal VERT");
        assert_eq!(p, PartitionType::Vert);
    }

    /// `infer_partition` returns `Split` when the leaf is smaller
    /// than any of `bsize` / `subsize(HORZ)` / `subsize(VERT)`. The
    /// recursion validates the sub-tree on the next call.
    #[test]
    fn infer_partition_split() {
        let mi_cols = 8usize;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; mi_cols * mi_cols];
        splat_cell(
            &mut grid,
            mi_cols,
            0,
            0,
            make_dc_cell(BlockSize::Block32x32),
        );
        let p = infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block64x64).expect("legal SPLIT");
        assert_eq!(p, PartitionType::Split);
    }

    /// A BLOCK_8x8 parent maps the three legal sub-8 leaf shapes onto
    /// its HORZ / VERT / SPLIT symbols without recursing below one mi.
    #[test]
    fn infer_partition_recognises_sub8_partitioning() {
        let mi_cols = 8usize;
        for (leaf, expected) in [
            (BlockSize::Block8x4, PartitionType::Horz),
            (BlockSize::Block4x8, PartitionType::Vert),
            (BlockSize::Block4x4, PartitionType::Split),
        ] {
            let mut grid: Vec<Option<ModeInfo>> = vec![None; mi_cols * mi_cols];
            grid[0] = Some(make_dc_cell(leaf));
            assert_eq!(
                infer_partition(&grid, mi_cols, 0, 0, BlockSize::Block8x8)
                    .expect("legal sub-8 partition"),
                expected
            );
        }
    }

    #[test]
    fn keyframe_block4x4_partition_and_bmi_modes_round_trip() {
        let fc = FrameContext::new_default();
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let expected_modes = [IntraMode::H, IntraMode::V, IntraMode::D45, IntraMode::Tm];
        let mut bmi = [BmiInfo::default(); 4];
        for (slot, mode) in bmi.iter_mut().zip(expected_modes) {
            slot.as_mode = mode as u8;
        }
        let sub8 = ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Tm,
                uv_mode: IntraMode::D135,
            },
            bmi,
        };
        assert_eq!(
            keyframe_above_block_mode(0, &[BmiInfo::default(); 4], Some(&sub8)),
            IntraMode::D45,
            "the block below a sub-8 leaf sees its bottom-left BMI mode"
        );
        assert_eq!(
            keyframe_left_block_mode(0, &[BmiInfo::default(); 4], Some(&sub8)),
            IntraMode::V,
            "the block right of a sub-8 leaf sees its top-right BMI mode"
        );
        for (index, expected) in expected_modes.into_iter().enumerate() {
            assert_eq!(
                keyframe_luma_mode_for_tx(
                    BlockSize::Block4x4,
                    &sub8.bmi,
                    IntraMode::Tm,
                    index & 1,
                    index >> 1,
                ),
                expected,
                "each 4x4 residual uses its own BMI transform type"
            );
        }
        let mut grid = vec![Some(make_dc_cell(BlockSize::Block8x8)); 64];
        grid[0] = Some(sub8);

        let mut buf = vec![0u8; 256];
        let written = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            let mut part_ctx = PartitionCtx::new(8);
            let mut nbr_ctx = NeighbourCtx::new(8);
            let mut coef_ctx = TileCoefCtx::new(8);
            let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
            write_partition_tree(
                &mut bw,
                &mut part_ctx,
                &mut nbr_ctx,
                &bounds,
                &grid,
                8,
                8,
                0,
                0,
                BlockSize::Block8x8,
                1,
                &fc,
                TxMode::AllowTx32x32,
                None,
                &mut coef_ctx,
                &coef_source,
                None,
            )
            .expect("write sub-8 keyframe leaf");
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(written);

        let mut reader = BoolReader::new(&buf).expect("reader");
        let partition = read_partition(&mut reader, &VP9_KF_PARTITION_PROBS[0], 0, None);
        assert_eq!(partition, PartitionType::Split);
        assert!(read_skip(&mut reader, &fc.skip_probs, 0, None));

        let mut decoded = [IntraMode::Dc; 4];
        for index in 0..4 {
            let above = if index < 2 {
                IntraMode::Dc
            } else {
                decoded[index - 2]
            };
            let left = if index == 0 || index == 2 {
                IntraMode::Dc
            } else {
                decoded[index - 1]
            };
            decoded[index] = read_intra_mode(
                &mut reader,
                &VP9_KF_Y_MODE_PROB[above as usize][left as usize],
            );
        }
        assert_eq!(decoded, expected_modes);
        assert_eq!(
            read_intra_mode(&mut reader, &VP9_KF_UV_MODE_PROB[IntraMode::Tm as usize],),
            IntraMode::D135
        );
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
        splat_cell(
            &mut grid,
            mi_cols as usize,
            0,
            0,
            make_dc_cell(BlockSize::Block32x64),
        );
        // (0, 4) claims Block16x16 instead of the Block32x64 a real
        // VERT would require.
        splat_cell(
            &mut grid,
            mi_cols as usize,
            0,
            4,
            make_dc_cell(BlockSize::Block16x16),
        );
        // Fill the bottom half so the writer doesn't trip on missing
        // cells first.
        splat_cell(
            &mut grid,
            mi_cols as usize,
            4,
            0,
            make_dc_cell(BlockSize::Block32x32),
        );
        splat_cell(
            &mut grid,
            mi_cols as usize,
            4,
            4,
            make_dc_cell(BlockSize::Block32x32),
        );

        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let fc = FrameContext::new_default();
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let err = write_tile_intra_keyframe_default(&mut bw, &grid, mi_rows, mi_cols, bounds, &fc)
            .unwrap_err();
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
        InterModeReadParams, get_intra_inter_context, get_mode_context, read_inter_block_mode_info,
        setup_compound_reference_mode,
    };
    use crate::inter::mv_ref::{BlockBounds, PrevFrameMvRef, sign_bias_from_header};
    use crate::mode_info::{BmiInfo, InterMode, Mv, RefFrame, read_is_inter, read_tx_size};
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
            seg_ref_frame: None,
            seg_skip: false,
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

    fn make_partial_bottom_inter_grid(mi_rows: u32, mi_cols: u32) -> Vec<Option<ModeInfo>> {
        assert_eq!((mi_rows, mi_cols), (7, 8));
        let mut grid = vec![None; (mi_rows * mi_cols) as usize];
        let mut splat = |row, col, bsize| {
            splat_cell(
                &mut grid,
                mi_cols as usize,
                row,
                col,
                make_inter_cell(
                    bsize,
                    InterMode::ZeroMv,
                    [RefFrame::LastFrame, RefFrame::NoRefFrame],
                    [Mv::ZERO; 2],
                    InterpFilter::EightTap,
                ),
            );
        };
        for col in [0, 4] {
            splat(0, col, BlockSize::Block32x32);
        }
        for col in [0, 2, 4, 6] {
            splat(4, col, BlockSize::Block16x16);
        }
        for col in 0..8 {
            splat(6, col, BlockSize::Block8x8);
        }
        grid
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
        if mi_row >= mi_rows || mi_col >= mi_cols {
            return;
        }
        let n8x8_l2 = n4x4_l2.saturating_sub(1);
        let num_8x8_wh: usize = 1usize << n8x8_l2;
        let hbs = num_8x8_wh >> 1;
        let has_rows = mi_row + hbs < mi_rows;
        let has_cols = mi_col + hbs < mi_cols;
        let mi_row_in_tile = mi_row - bounds.mi_row_start;
        let mi_col_in_tile = mi_col - bounds.mi_col_start;
        let bsl = n8x8_l2 as usize;
        let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);
        let probs = &fc.partition_prob[ctx];
        let partition = if has_rows && has_cols {
            read_partition(r, probs, ctx, None)
        } else if !has_rows && has_cols {
            if r.read(probs[1]) != 0 {
                PartitionType::Split
            } else {
                PartitionType::Horz
            }
        } else if has_rows && !has_cols {
            if r.read(probs[2]) != 0 {
                PartitionType::Split
            } else {
                PartitionType::Vert
            }
        } else {
            PartitionType::Split
        };
        let sub = subsize(bsize, partition).expect("legal subsize");

        let read_leaf = |r: &mut BoolReader<'_>,
                         leaves: &mut Vec<(usize, usize, ModeInfo)>,
                         mi_row: usize,
                         mi_col: usize,
                         leaf_bsize: BlockSize| {
            // Build the neighbour lookup the writer used.
            let nb = MiGridNeighbourLookup::new(mi_grid, mi_cols, mi_row, mi_col, bounds);
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
            let tx_size =
                read_tx_size(r, &fc.tx_probs, tx_mode, max_tx, allow_select, tx_ctx, None);
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
            let prev_mv = prev_mv_grid.and_then(|g| g.get(mi_row * mi_cols + mi_col).copied());
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
                r,
                fc,
                None,
                None,
                params,
                &nb,
                leaf_bsize,
                &bounds_block,
                prev_mv,
                &mut got,
            );
            // Avoid an unused-variable lint when callers don't read it.
            let _ = get_mode_context(leaf_bsize, &nb);
            leaves.push((mi_row, mi_col, got));
        };

        // BLOCK_8x8 parent (hbs == 0): non-NONE partitions produce a
        // single sub-8 leaf — mirrors the writer's special case +
        // the decoder's `hbs == 0` branch in `decode_inter_partition`.
        if hbs == 0 {
            read_leaf(r, leaves, mi_row, mi_col, sub);
            if (bsize as u8) >= (BlockSize::Block8x8 as u8) {
                part_ctx.update_with_subsize(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
            }
            return;
        }

        match partition {
            PartitionType::None => {
                read_leaf(r, leaves, mi_row, mi_col, sub);
            }
            PartitionType::Horz => {
                read_leaf(r, leaves, mi_row, mi_col, sub);
                if has_rows {
                    read_leaf(r, leaves, mi_row + hbs, mi_col, sub);
                }
            }
            PartitionType::Vert => {
                read_leaf(r, leaves, mi_row, mi_col, sub);
                if has_cols {
                    read_leaf(r, leaves, mi_row, mi_col + hbs, sub);
                }
            }
            PartitionType::Split => {
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
            let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
            let coef_source = crate::bitstream::AllZeroCoefSource::new();
            write_tile_inter(
                &mut bw,
                fc,
                params,
                tx_mode,
                grid,
                mi_rows,
                mi_cols,
                bounds,
                /* seg = */ None,
                prev_mv_grid,
                &mut coef_ctx,
                &coef_source,
                None,
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

    // -----------------------------------------------------------------
    // S4.16g phase 3 — non-skip inter leaves with coef-source
    // -----------------------------------------------------------------

    use crate::coef_ctx::{TileCoefCtx, combine_entropy_contexts};
    use crate::entropy::detokenize::decode_coefs;
    use crate::transforms::scans::scan_for;

    /// One TX block's recovered coefficients + eob, keyed by
    /// `(mi_row, mi_col, plane, tx_x4_local, tx_y4_local)`. Some
    /// fields are populated for inspection by future tests but go
    /// unread today.
    #[allow(dead_code)]
    #[derive(Clone, Debug)]
    struct RecoveredTx {
        mi_row: usize,
        mi_col: usize,
        plane: usize,
        tx_x4_local: usize,
        tx_y4_local: usize,
        tx_size: TxSize,
        coefs: Vec<i32>,
        eob: usize,
    }

    /// Decode-side mirror of `write_inter_block_residual`. Walks the
    /// same per-plane × per-TX grid the writer emitted, calling
    /// `decode_coefs` against the same `TileCoefCtx` state. Used by the
    /// non-skip replay walker below.
    ///
    /// Uses an "identity" dequantiser per `tx_size` so the recovered
    /// coefficients exactly equal the raw values the writer emitted.
    /// This matches `round_trip_block_tokens` in
    /// `entropy::detokenize::tests`.
    #[allow(clippy::too_many_arguments)]
    fn replay_inter_block_residual(
        r: &mut BoolReader<'_>,
        fc: &FrameContext,
        coef_ctx: &mut TileCoefCtx,
        mi_row: usize,
        mi_col: usize,
        leaf_bsize: BlockSize,
        luma_tx: TxSize,
        out: &mut Vec<RecoveredTx>,
    ) {
        const SS_X: [usize; 3] = [0, 1, 1];
        const SS_Y: [usize; 3] = [0, 1, 1];
        let mi_col_in_tile = mi_col;
        let mi_row_in_sb = mi_row & MI_MASK;

        for plane in 0..3 {
            let ss_x = SS_X[plane];
            let ss_y = SS_Y[plane];
            let plane_tx = if plane == 0 {
                luma_tx
            } else {
                uv_plane_tx_size(leaf_bsize, luma_tx)
            };
            let plane_bsize = if plane == 0 {
                leaf_bsize
            } else {
                uv_plane_block_size(leaf_bsize)
            };
            let is_sub8 = (leaf_bsize as u8) < (BlockSize::Block8x8 as u8);
            let (plane_bw_4x4, plane_bh_4x4) = if is_sub8 {
                (2 >> ss_x, 2 >> ss_y)
            } else {
                (
                    plane_bsize.n_4x4_wide() as usize,
                    plane_bsize.n_4x4_high() as usize,
                )
            };
            let tx_step_4x4 = 1usize << (plane_tx as usize);
            let plane_x4_base = (mi_col_in_tile * 2) >> ss_x;
            let plane_y4_in_sb_base = (mi_row_in_sb * 2) >> ss_y;
            let n_span = 1usize << (plane_tx as usize);
            let plane_type = usize::from(plane > 0);
            let coef_probs = &fc.coef_probs[plane_tx as usize][plane_type][1]; // ref_type = inter
            let scan = scan_for(plane_tx, TxType::DctDct);
            let n = plane_tx.n_coeffs();
            // Identity dequant: (1, 1) for non-32x32; (2, 2) for 32x32
            // (mirrors `round_trip_block_tokens::identity_dq`).
            let dq = if plane_tx == TxSize::Tx32x32 {
                (2, 2)
            } else {
                (1, 1)
            };

            let mut tx_y4_local = 0usize;
            while tx_y4_local < plane_bh_4x4 {
                let mut tx_x4_local = 0usize;
                while tx_x4_local < plane_bw_4x4 {
                    let x4 = plane_x4_base + tx_x4_local;
                    let y4_in_sb = plane_y4_in_sb_base + tx_y4_local;
                    // OR-fold the spanned above/left cells (mirrors
                    // `write_block_tokens`'s inline fold).
                    let mut a_any: u8 = 0;
                    for i in 0..n_span {
                        if coef_ctx.above_coef(plane, x4 + i) != 0 {
                            a_any = 1;
                            break;
                        }
                    }
                    let mut l_any: u8 = 0;
                    for i in 0..n_span {
                        if coef_ctx.left_coef(plane, y4_in_sb + i) != 0 {
                            l_any = 1;
                            break;
                        }
                    }
                    let initial_pt = combine_entropy_contexts(a_any, l_any);
                    let mut dq_out = vec![0i32; n];
                    let recovered_eob = decode_coefs(
                        r,
                        coef_probs,
                        None,
                        &scan,
                        initial_pt,
                        plane_tx,
                        n, // seg_eob
                        dq,
                        &mut dq_out,
                    );
                    coef_ctx.set_coef_flags(plane, x4, y4_in_sb, n_span, n_span, recovered_eob > 0);
                    out.push(RecoveredTx {
                        mi_row,
                        mi_col,
                        plane,
                        tx_x4_local,
                        tx_y4_local,
                        tx_size: plane_tx,
                        coefs: dq_out,
                        eob: recovered_eob,
                    });
                    tx_x4_local += tx_step_4x4;
                }
                tx_y4_local += tx_step_4x4;
            }
        }
    }

    /// Replay variant of `replay_inter_partition_tree` that also walks
    /// each leaf's residual coefs (when `skip == false`). Returns the
    /// recovered `(leaves, residual TX blocks)`.
    #[allow(clippy::too_many_arguments)]
    fn replay_inter_partition_tree_with_coefs(
        r: &mut BoolReader<'_>,
        fc: &FrameContext,
        params: &InterModeReadParams,
        tx_mode: TxMode,
        part_ctx: &mut TestPartCtx,
        coef_ctx: &mut TileCoefCtx,
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
        residuals: &mut Vec<RecoveredTx>,
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
                         coef_ctx: &mut TileCoefCtx,
                         leaves: &mut Vec<(usize, usize, ModeInfo)>,
                         residuals: &mut Vec<RecoveredTx>,
                         mi_row: usize,
                         mi_col: usize,
                         leaf_bsize: BlockSize| {
            let nb = MiGridNeighbourLookup::new(mi_grid, mi_cols, mi_row, mi_col, bounds);
            let above = nb.get(Position { row: -1, col: 0 }).copied();
            let left = nb.get(Position { row: 0, col: -1 }).copied();
            let skip_ctx = usize::from(above.is_some_and(|m| m.skip))
                + usize::from(left.is_some_and(|m| m.skip));
            let skip = read_skip(r, &fc.skip_probs, skip_ctx, None);
            let intra_inter_ctx = get_intra_inter_context(above.as_ref(), left.as_ref());
            let is_inter = read_is_inter(r, &fc.intra_inter_prob, intra_inter_ctx, None);
            assert!(is_inter, "is_inter @ ({mi_row}, {mi_col})");
            let allow_select = !skip;
            let max_tx = max_txsize(leaf_bsize);
            let above_tx = above.map(|m| (m.tx_size, m.skip));
            let left_tx = left.map(|m| (m.tx_size, m.skip));
            let tx_ctx = if max_tx >= TxSize::Tx8x8 {
                get_tx_size_context(above_tx, left_tx, max_tx)
            } else {
                0
            };
            let tx_size =
                read_tx_size(r, &fc.tx_probs, tx_mode, max_tx, allow_select, tx_ctx, None);
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
                r,
                fc,
                None,
                None,
                params,
                &nb,
                leaf_bsize,
                &bounds_block,
                prev_mv,
                &mut got,
            );
            let _ = get_mode_context(leaf_bsize, &nb);
            // Effective tx_size when allow_select is false matches
            // the writer's derivation (min(max_tx, cap)).
            let effective_tx = if allow_select && tx_mode == TxMode::TxModeSelect {
                tx_size
            } else {
                let cap = tx_mode_to_biggest_tx_size(tx_mode) as u8;
                let raw = core::cmp::min(max_tx as u8, cap);
                match raw {
                    0 => TxSize::Tx4x4,
                    1 => TxSize::Tx8x8,
                    2 => TxSize::Tx16x16,
                    3 => TxSize::Tx32x32,
                    _ => unreachable!(),
                }
            };
            if !skip {
                replay_inter_block_residual(
                    r,
                    fc,
                    coef_ctx,
                    mi_row,
                    mi_col,
                    leaf_bsize,
                    effective_tx,
                    residuals,
                );
            }
            leaves.push((mi_row, mi_col, got));
        };

        if hbs == 0 {
            read_leaf(r, coef_ctx, leaves, residuals, mi_row, mi_col, sub);
            if (bsize as u8) >= (BlockSize::Block8x8 as u8) {
                part_ctx.update_with_subsize(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
            }
            return;
        }

        match partition {
            PartitionType::None => {
                read_leaf(r, coef_ctx, leaves, residuals, mi_row, mi_col, sub);
            }
            PartitionType::Horz => {
                read_leaf(r, coef_ctx, leaves, residuals, mi_row, mi_col, sub);
                read_leaf(r, coef_ctx, leaves, residuals, mi_row + hbs, mi_col, sub);
            }
            PartitionType::Vert => {
                read_leaf(r, coef_ctx, leaves, residuals, mi_row, mi_col, sub);
                read_leaf(r, coef_ctx, leaves, residuals, mi_row, mi_col + hbs, sub);
            }
            PartitionType::Split => {
                replay_inter_partition_tree_with_coefs(
                    r,
                    fc,
                    params,
                    tx_mode,
                    part_ctx,
                    coef_ctx,
                    mi_grid,
                    mi_rows,
                    mi_cols,
                    bounds,
                    mi_row,
                    mi_col,
                    sub,
                    n8x8_l2,
                    prev_mv_grid,
                    leaves,
                    residuals,
                );
                replay_inter_partition_tree_with_coefs(
                    r,
                    fc,
                    params,
                    tx_mode,
                    part_ctx,
                    coef_ctx,
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
                    residuals,
                );
                replay_inter_partition_tree_with_coefs(
                    r,
                    fc,
                    params,
                    tx_mode,
                    part_ctx,
                    coef_ctx,
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
                    residuals,
                );
                replay_inter_partition_tree_with_coefs(
                    r,
                    fc,
                    params,
                    tx_mode,
                    part_ctx,
                    coef_ctx,
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
                    residuals,
                );
            }
        }

        if (bsize as u8) >= (BlockSize::Block8x8 as u8)
            && (bsize == BlockSize::Block8x8 || partition != PartitionType::Split)
        {
            part_ctx.update_with_subsize(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
        }
    }

    /// Encode `grid` via `write_tile_inter` against `coef_source`, then
    /// replay through the per-symbol readers including per-TX coef
    /// decode. Returns the recovered leaves; residuals can be inspected
    /// via [`encode_then_replay_inter_full`].
    #[allow(clippy::too_many_arguments)]
    fn encode_then_replay_inter_with_coefs(
        fc: &FrameContext,
        params: &InterModeReadParams,
        tx_mode: TxMode,
        mi_rows: u32,
        mi_cols: u32,
        grid: &[Option<ModeInfo>],
        prev_mv_grid: Option<&[PrevFrameMvRef]>,
        coef_source: &dyn InterCoefSource,
    ) -> Vec<(usize, usize, ModeInfo)> {
        let (leaves, _residuals) = encode_then_replay_inter_full(
            fc,
            params,
            tx_mode,
            mi_rows,
            mi_cols,
            grid,
            prev_mv_grid,
            coef_source,
        );
        leaves
    }

    /// Full-fidelity variant of `encode_then_replay_inter_with_coefs`
    /// that also returns the recovered residual TX blocks.
    #[allow(clippy::too_many_arguments)]
    fn encode_then_replay_inter_full(
        fc: &FrameContext,
        params: &InterModeReadParams,
        tx_mode: TxMode,
        mi_rows: u32,
        mi_cols: u32,
        grid: &[Option<ModeInfo>],
        prev_mv_grid: Option<&[PrevFrameMvRef]>,
        coef_source: &dyn InterCoefSource,
    ) -> (Vec<(usize, usize, ModeInfo)>, Vec<RecoveredTx>) {
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        // Large buffer to fit dense residuals (every TX position
        // potentially several bits of token tree).
        let mut buf = vec![0u8; 128 * 1024];
        let written = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
            write_tile_inter(
                &mut bw,
                fc,
                params,
                tx_mode,
                grid,
                mi_rows,
                mi_cols,
                bounds,
                /* seg = */ None,
                prev_mv_grid,
                &mut coef_ctx,
                coef_source,
                None,
            )
            .expect("write_tile_inter should accept legal grid");
            for _ in 0..32 {
                bw.write_bit(0);
            }
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(written);

        let mut r = BoolReader::new(&buf).expect("reader init");
        let mut part_ctx = TestPartCtx::new(mi_cols as usize);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let mut leaves = Vec::new();
        let mut residuals = Vec::new();
        let mut mi_row = 0usize;
        while mi_row < mi_rows as usize {
            part_ctx.reset_left();
            coef_ctx.reset_left_coef();
            let mut mi_col = 0usize;
            while mi_col < mi_cols as usize {
                replay_inter_partition_tree_with_coefs(
                    &mut r,
                    fc,
                    params,
                    tx_mode,
                    &mut part_ctx,
                    &mut coef_ctx,
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
                    &mut residuals,
                );
                mi_col += MI_BLOCK_SIZE;
            }
            mi_row += MI_BLOCK_SIZE;
        }
        (leaves, residuals)
    }

    /// S4.16g phase 3a — round-trip a single 64×64 inter leaf with
    /// `skip == false` and an `AllZeroCoefSource`. Proves the writer's
    /// coef-walk plumbs through (4 Y TX + 1 U TX + 1 V TX = 6 TX
    /// blocks emit a terminal-EOB each) and the in-test replay walker
    /// recovers the same leaves the writer wrote.
    #[test]
    fn s4_16g_round_trip_non_skip_all_zero_coefs() {
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

        let (leaves, residuals) = encode_then_replay_inter_full(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
            &crate::bitstream::AllZeroCoefSource::new(),
        );
        assert_eq!(leaves.len(), 1, "1 leaf");
        assert_inter_match(&cell, &leaves[0].2, "non-skip leaf");
        // BLOCK_64x64 with effective_tx = Tx32x32:
        //   Y: 2×2 TX blocks (64/4 = 16 4x4, step 8 = 2 along each axis)
        //   U: 1 TX block (uv_bsize = Block32x32, plane_tx = Tx32x32, 8 cells, step 8)
        //   V: 1 TX block
        // ⇒ 4 + 1 + 1 = 6 TX blocks total.
        assert_eq!(residuals.len(), 6, "1 leaf × (4 Y + 1 U + 1 V) TX blocks");
        for tx in &residuals {
            assert_eq!(tx.eob, 0, "all-zero source ⇒ eob=0");
            assert!(
                tx.coefs.iter().all(|&v| v == 0),
                "all-zero source ⇒ coefs all 0"
            );
        }
    }

    /// S4.16g phase 3a — round-trip a non-skip inter leaf where the
    /// Y plane carries a single non-zero coefficient (`coefs[0] = 1`,
    /// eob = 1) per Y TX block. UV planes stay all-zero. Proves the
    /// coef walk actually FIRES (versus being silently bypassed by
    /// the all-zero source) and that per-TX coef recovery agrees
    /// with what the writer emitted.
    #[test]
    fn s4_16g_round_trip_non_skip_one_per_y_coef() {
        struct OnePerYCoefSource {
            zero: [i16; 1024],
            one_y: [i16; 1024],
        }
        impl OnePerYCoefSource {
            fn new() -> Self {
                let mut one_y = [0i16; 1024];
                // DC (raster position 0) = 1. The default scan tables
                // map scan position 0 → raster position 0, so this is
                // the DC-only block.
                one_y[0] = 1;
                Self {
                    zero: [0; 1024],
                    one_y,
                }
            }
        }
        impl InterCoefSource for OnePerYCoefSource {
            fn get_block_coefs(
                &self,
                _mi_row: usize,
                _mi_col: usize,
                plane: usize,
                _tx_x4_local: usize,
                _tx_y4_local: usize,
                tx_size: TxSize,
            ) -> (&[i16], usize) {
                if plane == 0 {
                    (&self.one_y[..tx_size.n_coeffs()], 1)
                } else {
                    (&self.zero[..tx_size.n_coeffs()], 0)
                }
            }
        }

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

        let src = OnePerYCoefSource::new();
        let (leaves, residuals) = encode_then_replay_inter_full(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
            &src,
        );
        assert_eq!(leaves.len(), 1, "1 leaf");
        assert_inter_match(&cell, &leaves[0].2, "non-skip 1Y leaf");
        assert_eq!(residuals.len(), 6, "1 leaf × 6 TX blocks");
        for tx in &residuals {
            if tx.plane == 0 {
                assert_eq!(tx.eob, 1, "Y TX: eob == 1");
                assert_eq!(tx.coefs[0], 1, "Y TX: DC coef recovered as 1");
                // Decode side uses identity dequant; positions past
                // scan[0] are zeroed by `decode_coefs`.
                for (i, &v) in tx.coefs.iter().enumerate().skip(1) {
                    assert_eq!(v, 0, "Y TX: scan pos {i} should be 0");
                }
            } else {
                assert_eq!(tx.eob, 0, "UV TX: eob == 0");
                assert!(tx.coefs.iter().all(|&v| v == 0), "UV TX: coefs all 0");
            }
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

    // ----- S4.16f: sub-8 partitioning at BLOCK_8x8 parent ----------

    /// Build an 8x8-grain inter cell with sub-8 partitioning recorded
    /// via the [`BmiInfo`] array. All sub-blocks ZEROMV LASTFRAME,
    /// skip=true. The parent's `mode` / `mvs` follow libvpx's "last
    /// sub-block becomes parent" convention.
    fn make_sub8_zero_inter_cell(sub8: BlockSize) -> ModeInfo {
        let mut bmi = [BmiInfo::default(); 4];
        for slot in &mut bmi {
            slot.as_mode = InterMode::ZeroMv.idx() as u8;
            slot.as_mv = [Mv::ZERO; 2];
        }
        ModeInfo {
            block_size: sub8,
            tx_size: TxSize::Tx4x4,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi,
        }
    }

    /// 64×64 SB drilled all the way down to 8x8 leaves, with one
    /// `Block4x4` sub-8 leaf at (0,0). Proves the partition walker
    /// reaches sub-8 via the `hbs == 0` branch (S4.16f) and dispatches
    /// to `write_sub8_bmi` (S4.16e).
    #[test]
    fn s4_16f_round_trip_block8x8_split_into_4x4_at_corner() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;

        // Fill the 64x64 with Block8x8 NONE leaves...
        let plain_cell = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut grid: Vec<Option<ModeInfo>> = vec![Some(plain_cell); (mi_rows * mi_cols) as usize];

        // ...then replace cell (0, 0) with a Block4x4 sub-8 leaf.
        grid[0] = Some(make_sub8_zero_inter_cell(BlockSize::Block4x4));

        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );

        // 64 leaves total — 63 Block8x8 + 1 Block4x4.
        assert_eq!(leaves.len(), 64);
        let (r0, c0, mi0) = &leaves[0];
        assert_eq!((*r0, *c0), (0, 0));
        assert_eq!(
            mi0.block_size,
            BlockSize::Block4x4,
            "sub-8 leaf at (0,0) must round-trip as Block4x4"
        );
        // Spot-check one of the surrounding Block8x8 leaves.
        let (_, _, mi1) = &leaves[1];
        assert_eq!(mi1.block_size, BlockSize::Block8x8);
    }

    /// 64x64 SB with one `Block8x4` sub-8 leaf (HORZ at BLOCK_8x8) at (0,0).
    #[test]
    fn s4_16f_round_trip_block8x8_horz_into_8x4_at_corner() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let plain_cell = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut grid: Vec<Option<ModeInfo>> = vec![Some(plain_cell); (mi_rows * mi_cols) as usize];
        grid[0] = Some(make_sub8_zero_inter_cell(BlockSize::Block8x4));

        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 64);
        assert_eq!(leaves[0].2.block_size, BlockSize::Block8x4);
    }

    /// 64x64 SB with one `Block4x8` sub-8 leaf (VERT at BLOCK_8x8) at (0,0).
    #[test]
    fn s4_16f_round_trip_block8x8_vert_into_4x8_at_corner() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let plain_cell = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut grid: Vec<Option<ModeInfo>> = vec![Some(plain_cell); (mi_rows * mi_cols) as usize];
        grid[0] = Some(make_sub8_zero_inter_cell(BlockSize::Block4x8));

        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 64);
        assert_eq!(leaves[0].2.block_size, BlockSize::Block4x8);
    }

    /// Phase 2b: Leaf carrying `BlockMode::Intra` is now accepted
    /// by the inter-frame writer — the intra-in-inter path writes
    /// `is_inter=0`, y_mode, and uv_mode.
    #[test]
    fn phase_2b_accepts_intra_leaf() {
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
        cell.skip = true;
        let grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let result = write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            /* seg = */ None,
            /* prev_mv_grid = */ None,
            &mut coef_ctx,
            &coef_source,
            None,
        );
        assert!(result.is_ok(), "intra-in-inter should succeed: {result:?}");
    }

    /// S4.16g phase 3: the inter writer no longer rejects `skip =
    /// false`. With an `AllZeroCoefSource` plumbed in, the writer
    /// emits the residual path's terminal-EOB per TX block and the
    /// round-trip succeeds. Replaces the original
    /// `s4_16c_rejects_non_skip_leaf` (which became obsolete the
    /// moment the writer learned to emit residuals).
    #[test]
    fn s4_16g_accepts_non_skip_leaf_with_all_zero_coefs() {
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
        let leaves = encode_then_replay_inter_with_coefs(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
            &crate::bitstream::AllZeroCoefSource::new(),
        );
        assert_eq!(leaves.len(), 1);
        assert_inter_match(&cell, &leaves[0].2, "non-skip leaf all-zero coefs");
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
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let err = write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            /* seg = */ None,
            /* prev_mv_grid = */ None,
            &mut coef_ctx,
            &coef_source,
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
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let err = write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            /* seg = */ None,
            /* prev_mv_grid = */ None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                ModesWriterError::MissingCell {
                    mi_row: 0,
                    mi_col: 0
                }
            ),
            "got: {err:?}"
        );
    }

    /// A seven-mi-row inter frame round-trips through the decoder's
    /// reduced bottom-edge partition syntax.
    #[test]
    fn s4_16c_round_trips_partial_bottom_sb() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 7u32;
        let mi_cols = 8u32;
        let grid = make_partial_bottom_inter_grid(mi_rows, mi_cols);
        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 14);
        assert!(leaves.iter().all(|(row, _, mi)| {
            row + usize::from(mi.block_size.n_8x8_high()) <= mi_rows as usize
        }));
    }

    // ---------------------------------------------------------------
    // AQ.0 — segmentation-map emission (encoder side)
    //
    // The decoder already supports the full per-block segment_id
    // read path (`mode_info::read_segment_id` + `read_inter_segment_id`).
    // These tests drive `write_segment_id` through the keyframe + inter
    // writer scopes and replay via the same `BoolReader` flow the
    // decoder uses, asserting the recovered segment_id matches the
    // input. v1 scope is `update_map = true, temporal_update = false`
    // (cyclic refresh AQ — libvpx mode 3).
    // ---------------------------------------------------------------

    /// AQ.0 — write a single keyframe leaf with `segment_id = 3` and
    /// recover it via `read_segment_id`. Exercises the writer's new
    /// `Some(SegmentationWriteParams)` branch on the KF path; mirrors
    /// the decoder's `params.segmentation_update_map` branch in
    /// `decode_block`.
    #[test]
    fn write_block_intra_keyframe_emits_nonzero_segment_id() {
        use crate::bitstream::SEG_TREE_PROBS;
        let mut buf = vec![0u8; 64];
        let fc = FrameContext::new_default();
        let tree_probs = [128u8; SEG_TREE_PROBS];
        let n = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            let bounds = TileBounds {
                mi_row_start: 0,
                mi_row_end: 8,
                mi_col_start: 0,
                mi_col_end: 8,
            };
            let mut nbr_ctx = NeighbourCtx::new(8);
            let mut grid = make_dc_grid(8, 8);
            for cell in grid.iter_mut().flatten() {
                cell.segment_id = 3;
            }
            let mi = grid[0].expect("populated cell");
            let mut coef_ctx = TileCoefCtx::new(8);
            let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
            write_block_intra_keyframe(
                &mut bw,
                &fc,
                TxMode::AllowTx32x32,
                Some(SegmentationWriteParams {
                    tree_probs: &tree_probs,
                }),
                &mut nbr_ctx,
                &bounds,
                &grid,
                8,
                8,
                0,
                0,
                BlockSize::Block64x64,
                &mi,
                &mut coef_ctx,
                &coef_source,
                None,
            )
            .expect("write_block_intra_keyframe with seg should succeed");
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(n);

        // Replay: segment_id is the FIRST symbol on the wire, then
        // skip, y_mode, uv_mode.
        let mut reader = BoolReader::new(&buf).expect("reader init");
        let got_seg = crate::mode_info::read_segment_id(&mut reader, &tree_probs);
        assert_eq!(got_seg, 3, "segment_id should round-trip");
        let skip = read_skip(&mut reader, &fc.skip_probs, 0, None);
        assert!(skip);
        let y_probs = &VP9_KF_Y_MODE_PROB[IntraMode::Dc as usize][IntraMode::Dc as usize];
        let got_y = read_intra_mode(&mut reader, y_probs);
        assert_eq!(got_y, IntraMode::Dc);
        let uv_probs = &VP9_KF_UV_MODE_PROB[got_y as usize];
        let got_uv = read_intra_mode(&mut reader, uv_probs);
        assert_eq!(got_uv, IntraMode::Dc);
    }

    /// AQ.0 — keyframe writer round-trips every legal segment_id
    /// (0..=7) at p=128 via a 1-SB grid through the FULL
    /// `write_tile_intra_keyframe` entry point. Exercises the
    /// partition-tree path AND the segment_id emission.
    #[test]
    fn write_tile_intra_keyframe_round_trip_segment_ids_0_to_7() {
        use crate::bitstream::SEG_TREE_PROBS;
        let fc = FrameContext::new_default();
        let tree_probs = [128u8; SEG_TREE_PROBS];

        for target_seg in 0u8..(crate::bitstream::MAX_SEGMENTS as u8) {
            let mi_rows = 8u32;
            let mi_cols = 8u32;
            let mut grid = make_dc_grid(mi_rows as usize, mi_cols as usize);
            for mi in grid.iter_mut().flatten() {
                mi.segment_id = target_seg;
            }
            let bounds = TileBounds {
                mi_row_start: 0,
                mi_row_end: 8,
                mi_col_start: 0,
                mi_col_end: 8,
            };
            let mut buf = vec![0u8; 1024];
            let written = {
                let mut bw = BoolWriter::start_encode(&mut buf);
                let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
                let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
                write_tile_intra_keyframe(
                    &mut bw,
                    &grid,
                    mi_rows,
                    mi_cols,
                    bounds,
                    &fc,
                    TxMode::AllowTx32x32,
                    Some(SegmentationWriteParams {
                        tree_probs: &tree_probs,
                    }),
                    &mut coef_ctx,
                    &coef_source,
                )
                .expect("write_tile_intra_keyframe with seg");
                bw.stop_encode().expect("stop_encode")
            };
            buf.truncate(written);

            // Walk the same per-symbol flow the decoder uses:
            // partition → segment_id → skip → y_mode → uv_mode.
            let mut r = BoolReader::new(&buf).expect("BoolReader::new");
            // 64x64 SB → partition_ctx = bsl*4 = 3*4 = 12 for the empty
            // above/left state.
            let part_ctx = 3 * 4;
            let p = read_partition(&mut r, &VP9_KF_PARTITION_PROBS[part_ctx], part_ctx, None);
            assert_eq!(p, PartitionType::None, "seg={target_seg}: partition");
            let got_seg = crate::mode_info::read_segment_id(&mut r, &tree_probs);
            assert_eq!(
                got_seg, target_seg,
                "seg={target_seg}: recovered segment_id"
            );
            let skip = read_skip(&mut r, &fc.skip_probs, 0, None);
            assert!(skip, "seg={target_seg}: skip");
        }
    }

    /// AQ.0 — multi-segment 64×64 SB split into 4 32×32 SPLIT
    /// leaves with `segment_id` values [0, 1, 2, 3]. Verifies the
    /// recursion threads the segmentation context to every leaf
    /// and each leaf emits its own tree-decoded segment_id.
    #[test]
    fn write_tile_intra_keyframe_multi_segment_split_sb() {
        use crate::bitstream::SEG_TREE_PROBS;
        let fc = FrameContext::new_default();
        let tree_probs = [128u8; SEG_TREE_PROBS];
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        // 4 leaves of Block32x32 (4 mi_units each on each side).
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let segs_at = [
            (0usize, 0usize, 0u8), // top-left
            (0, 4, 1),             // top-right
            (4, 0, 2),             // bottom-left
            (4, 4, 3),             // bottom-right
        ];
        for &(r, c, seg) in &segs_at {
            let mut cell = make_dc_cell(BlockSize::Block32x32);
            cell.segment_id = seg;
            splat_cell(&mut grid, mi_cols as usize, r, c, cell);
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 2 * 1024];
        let written = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
            let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &grid,
                mi_rows,
                mi_cols,
                bounds,
                &fc,
                TxMode::AllowTx32x32,
                Some(SegmentationWriteParams {
                    tree_probs: &tree_probs,
                }),
                &mut coef_ctx,
                &coef_source,
            )
            .expect("write_tile_intra_keyframe SPLIT with seg");
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(written);

        // Replay: parent 64x64 → PARTITION_SPLIT (1, 1, 1) at
        // ctx=12. Each child 32x32 then emits PARTITION_NONE (a
        // single 0 bit at the appropriate ctx). At each leaf we read
        // segment_id → skip → y_mode → uv_mode.
        let mut r = BoolReader::new(&buf).expect("BoolReader::new");
        let mut test_ctx = TestPartCtx::new(mi_cols as usize);

        // Parent SPLIT.
        let parent_ctx = test_ctx.context(0, 0, 3);
        let parent_probs = &VP9_KF_PARTITION_PROBS[parent_ctx];
        let parent = read_partition(&mut r, parent_probs, parent_ctx, None);
        assert_eq!(parent, PartitionType::Split, "parent partition");

        // 4 children at (0,0), (0,4), (4,0), (4,4), each 32x32 with
        // PARTITION_NONE.
        for &(rr, cc, expected_seg) in &segs_at {
            // For 32x32 inside a 64x64 split, bsl = n8x8_l2 = 2.
            let child_ctx = test_ctx.context(rr, cc, 2);
            let child_probs = &VP9_KF_PARTITION_PROBS[child_ctx];
            let p = read_partition(&mut r, child_probs, child_ctx, None);
            assert_eq!(p, PartitionType::None, "child ({rr},{cc})");
            test_ctx.update(rr, cc, BlockSize::Block32x32.n_8x8_wide() as usize);

            // Segment_id.
            let got_seg = crate::mode_info::read_segment_id(&mut r, &tree_probs);
            assert_eq!(got_seg, expected_seg, "child ({rr},{cc}) segment_id");

            // Skip: skip_ctx depends on above/left neighbour skip flags.
            // Every leaf is skip=true; so e.g. top-right sees left=skip=1.
            let skip_ctx = match (rr, cc) {
                (0, 0) => 0,
                (0, 4) | (4, 0) => 1,
                (4, 4) => 2,
                _ => unreachable!(),
            };
            let skip = read_skip(&mut r, &fc.skip_probs, skip_ctx, None);
            assert!(skip, "child ({rr},{cc}) skip");
            let y_probs = &VP9_KF_Y_MODE_PROB[IntraMode::Dc as usize][IntraMode::Dc as usize];
            let got_y = read_intra_mode(&mut r, y_probs);
            assert_eq!(got_y, IntraMode::Dc, "child ({rr},{cc}) y_mode");
            let uv_probs = &VP9_KF_UV_MODE_PROB[got_y as usize];
            let got_uv = read_intra_mode(&mut r, uv_probs);
            assert_eq!(got_uv, IntraMode::Dc, "child ({rr},{cc}) uv_mode");
        }
    }

    /// AQ.0 — inter writer emits per-leaf `segment_id` via tree
    /// decode (the `update_map && !temporal_update` branch in
    /// `read_inter_segment_id`). Drives a single Block64x64 inter
    /// leaf with `segment_id = 2`; recovered via per-symbol replay.
    #[test]
    fn write_tile_inter_emits_nonzero_segment_id() {
        use crate::bitstream::SEG_TREE_PROBS;
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let tree_probs = [128u8; SEG_TREE_PROBS];

        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.segment_id = 2;
        let grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let written = {
            let mut bw = BoolWriter::start_encode(&mut buf);
            let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
            let coef_source = crate::bitstream::AllZeroCoefSource::new();
            write_tile_inter(
                &mut bw,
                &fc,
                &params,
                TxMode::AllowTx32x32,
                &grid,
                mi_rows,
                mi_cols,
                bounds,
                Some(SegmentationWriteParams {
                    tree_probs: &tree_probs,
                }),
                /* prev_mv_grid = */ None,
                &mut coef_ctx,
                &coef_source,
                None,
            )
            .expect("write_tile_inter with seg");
            for _ in 0..16 {
                bw.write_bit(0);
            }
            bw.stop_encode().expect("stop_encode")
        };
        buf.truncate(written);

        let mut r = BoolReader::new(&buf).expect("BoolReader::new");
        let part_ctx = 3 * 4;
        let part_probs = &fc.partition_prob[part_ctx];
        let p = read_partition(&mut r, part_probs, part_ctx, None);
        assert_eq!(p, PartitionType::None);
        let got_seg = crate::mode_info::read_segment_id(&mut r, &tree_probs);
        assert_eq!(got_seg, 2, "inter leaf segment_id should round-trip");
        let skip = read_skip(&mut r, &fc.skip_probs, 0, None);
        assert!(skip);
        let intra_inter_ctx = 0;
        let is_inter = read_is_inter(&mut r, &fc.intra_inter_prob, intra_inter_ctx, None);
        assert!(is_inter);
    }

    /// AQ.0 — writer rejects `seg_id_predicted == true`. The
    /// temporal-update predicted-flag bit is out of scope for v1.
    #[test]
    fn write_tile_inter_rejects_seg_id_predicted() {
        use crate::bitstream::SEG_TREE_PROBS;
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let tree_probs = [128u8; SEG_TREE_PROBS];

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
        cell.seg_id_predicted = true;
        let grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let err = write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            Some(SegmentationWriteParams {
                tree_probs: &tree_probs,
            }),
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, ModesWriterError::UnsupportedSegmentation { .. }),
            "got: {err:?}"
        );
    }

    /// AQ.0 — writer accepts the boundary segment_id (MAX_SEGMENTS-1
    /// = 7). Edge-of-range sanity check matching libvpx's
    /// `MAX_SEGMENTS = 8` constant.
    #[test]
    fn write_tile_intra_keyframe_accepts_max_segment_id() {
        use crate::bitstream::SEG_TREE_PROBS;
        let fc = FrameContext::new_default();
        let tree_probs = [128u8; SEG_TREE_PROBS];
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid = make_dc_grid(mi_rows as usize, mi_cols as usize);
        for mi in grid.iter_mut().flatten() {
            mi.segment_id = (crate::bitstream::MAX_SEGMENTS as u8) - 1;
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut buf = vec![0u8; 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        write_tile_intra_keyframe(
            &mut bw,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            &fc,
            TxMode::AllowTx32x32,
            Some(SegmentationWriteParams {
                tree_probs: &tree_probs,
            }),
            &mut coef_ctx,
            &coef_source,
        )
        .expect("MAX_SEGMENTS-1 = 7 must be accepted");
    }

    // ---------- Stage 5 — branch-coverage tests -------------------------
    //
    // The tests below target the missed-branch dimensions in
    // `write_tile_inter` / `write_inter_block_leaf` /
    // `write_inter_partition_tree`. Coverage focus areas:
    //   - Compound reference encoding (Compound + Select).
    //   - Switchable interpolation filter encoding.
    //   - NEWMV at extreme magnitudes with HP precision.
    //   - All 4 MV joints (covered by varying `Mv` row/col combos).
    //   - Per-block-size inter round-trip (8x16 / 16x8 / 16x32 / 32x16 /
    //     32x64 / 64x32) exercising `infer_inter_partition` paths.

    /// Compound `[LAST, ALTREF]` ZEROMV round-trip via the tile-level
    /// inter writer. Exercises the compound write path of
    /// `write_inter_block_mode_info` from inside the partition walker.
    #[test]
    fn s5_tile_inter_compound_zero_mv_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
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
        assert_inter_match(&cell, &leaves[0].2, "compound LAST+ALTREF ZEROMV 64x64");
    }

    /// Compound `[LAST, ALTREF]` NEWMV round-trip via tile-level
    /// writer with per-ref MVs. Exercises both per-ref `write_mv`
    /// calls.
    #[test]
    fn s5_tile_inter_compound_new_mv_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        params.allow_high_precision_mv = true;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mvs = [Mv { row: 8, col: -4 }, Mv { row: -12, col: 6 }];
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
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
        assert_inter_match(&cell, &leaves[0].2, "compound LAST+ALTREF NEWMV");
    }

    /// Switchable interpolation filter round-trip via tile-level
    /// writer (EightTapSmooth on a 64x64 ZEROMV leaf).
    #[test]
    fn s5_tile_inter_switchable_filter_smooth_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.interp_filter = InterpFilter::Switchable;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSmooth,
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
        assert_inter_match(&cell, &leaves[0].2, "switchable smooth 64x64 ZEROMV");
    }

    /// Switchable interpolation filter — EightTapSharp variant.
    #[test]
    fn s5_tile_inter_switchable_filter_sharp_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.interp_filter = InterpFilter::Switchable;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSharp,
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
        assert_inter_match(&cell, &leaves[0].2, "switchable sharp 64x64 ZEROMV");
    }

    /// ReferenceMode::Select with a single-ref leaf — the writer
    /// emits the `comp_inter` bit = 0 then the single-ref bits.
    #[test]
    fn s5_tile_inter_reference_mode_select_single_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.reference_mode = ReferenceMode::Select;
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
        assert_inter_match(&cell, &leaves[0].2, "select+single");
    }

    /// ReferenceMode::Select with a compound leaf — emits the
    /// `comp_inter` bit = 1 then the compound bit.
    #[test]
    fn s5_tile_inter_reference_mode_select_compound_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Select;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
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
        assert_inter_match(&cell, &leaves[0].2, "select+compound");
    }

    /// Single-ref GOLDEN NEWMV: exercises the `bit0=1, bit1=0`
    /// single-ref encoder path.
    #[test]
    fn s5_tile_inter_single_golden_new_mv_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.allow_high_precision_mv = true;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::NewMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv { row: 4, col: -2 }, Mv::ZERO],
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
        assert_inter_match(&cell, &leaves[0].2, "single GOLDEN NEWMV");
    }

    /// Single-ref ALTREF NEWMV: exercises the `bit0=1, bit1=1`
    /// single-ref encoder path.
    #[test]
    fn s5_tile_inter_single_altref_new_mv_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.allow_high_precision_mv = true;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::NewMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv { row: -8, col: 4 }, Mv::ZERO],
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
        assert_inter_match(&cell, &leaves[0].2, "single ALTREF NEWMV");
    }

    /// MV joint HnzVnz (both axes nonzero) via NEWMV. The encoder's
    /// `write_mv` walks the joint tree to `HnzVnz` and emits both
    /// row and col components.
    #[test]
    fn s5_tile_inter_new_mv_both_axes_nonzero_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.allow_high_precision_mv = true;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv { row: 6, col: 10 }, Mv::ZERO],
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
        assert_inter_match(&cell, &leaves[0].2, "NEWMV both axes nonzero");
    }

    /// MV joint HnzVz (column nonzero, row zero) via NEWMV.
    #[test]
    fn s5_tile_inter_new_mv_col_only_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.allow_high_precision_mv = true;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv { row: 0, col: 14 }, Mv::ZERO],
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
        assert_inter_match(&cell, &leaves[0].2, "NEWMV col only");
    }

    /// MV joint Zero via NEWMV (encoder still emits one `write_mv`
    /// call; the joint tree picks the Zero leaf and no per-axis bits
    /// follow).
    #[test]
    fn s5_tile_inter_new_mv_zero_diff_round_trip() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        // NEWMV with target MV equal to ref MV (= 0 for no
        // neighbours) → diff = 0 → joint = Zero.
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::NewMv,
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
        assert_inter_match(&cell, &leaves[0].2, "NEWMV diff = Zero joint");
    }

    /// Per-block-size partition: 32x32 NEWMV (SPLIT into 4× 32x32).
    /// One quadrant uses HP-allowed precision; all share a single
    /// neighbour shape.
    #[test]
    fn s5_tile_inter_block32x32_new_mv_split_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.allow_high_precision_mv = true;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];

        let make_quad = |mv: Mv| {
            make_inter_cell(
                BlockSize::Block32x32,
                InterMode::NewMv,
                [RefFrame::LastFrame, RefFrame::NoRefFrame],
                [mv, Mv::ZERO],
                InterpFilter::EightTap,
            )
        };
        let cell_tl = make_quad(Mv { row: 5, col: 3 });
        let cell_tr = make_quad(Mv { row: -7, col: 11 });
        let cell_bl = make_quad(Mv { row: 13, col: -5 });
        let cell_br = make_quad(Mv { row: -9, col: -15 });
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
        assert_eq!(leaves.len(), 4);
        let expected = [&cell_tl, &cell_tr, &cell_bl, &cell_br];
        for (i, ec) in expected.iter().enumerate() {
            assert_inter_match(ec, &leaves[i].2, "32x32 NEWMV SPLIT quadrant");
        }
    }

    /// 16x16 inter blocks tiled across 32x32 SPLIT (4 quadrants × 4
    /// per quadrant = 16 leaves). Stress the partition walker's
    /// recursion past 32x32 → 16x16.
    #[test]
    fn s5_tile_inter_block16x16_dense_round_trip() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = make_inter_cell(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // 4 mi-cols per 16x16 → 2x2 = 4 leaves per 32x32 quadrant.
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(2) {
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
        assert_eq!(leaves.len(), 16, "16x16 grid: 16 leaves");
        for (_, _, mi) in &leaves {
            assert_eq!(mi.block_size, BlockSize::Block16x16);
        }
    }

    /// 8x16 inter block via HORZ→VERT partition split. Exercises the
    /// `infer_inter_partition` VERT path at 16x16 → 8x16.
    #[test]
    fn s5_tile_inter_block8x16_vert_round_trip() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // Top-left 16x16 region: 2x VERT → two 8x16 leaves.
        let left = make_inter_cell(
            BlockSize::Block8x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, left);
        splat_cell(&mut grid, mi_cols as usize, 0, 1, left);
        // Fill remainder with a single 16x16 ZEROMV stamp.
        let stamp = make_inter_cell(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(2) {
                if r == 0 && c == 0 {
                    continue;
                }
                splat_cell(&mut grid, mi_cols as usize, r, c, stamp);
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
        // 2 (8x16) + 15 (16x16) = 17 leaves.
        assert_eq!(leaves.len(), 17);
        // First two leaves are the 8x16 pair.
        assert_eq!(leaves[0].2.block_size, BlockSize::Block8x16);
        assert_eq!(leaves[1].2.block_size, BlockSize::Block8x16);
    }

    /// 16x8 inter block via HORZ split. Exercises the HORZ path at
    /// 16x16 → 16x8.
    #[test]
    fn s5_tile_inter_block16x8_horz_round_trip() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let top = make_inter_cell(
            BlockSize::Block16x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, top);
        splat_cell(&mut grid, mi_cols as usize, 1, 0, top);
        // Fill remainder with 16x16.
        let stamp = make_inter_cell(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(2) {
                if r == 0 && c == 0 {
                    continue;
                }
                splat_cell(&mut grid, mi_cols as usize, r, c, stamp);
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
        assert_eq!(leaves.len(), 17);
        assert_eq!(leaves[0].2.block_size, BlockSize::Block16x8);
        assert_eq!(leaves[1].2.block_size, BlockSize::Block16x8);
    }

    /// 32x64 inter block (full-tile VERT at root). Exercises the
    /// inter partition writer's VERT case at the 64x64 → 32x64 level.
    #[test]
    fn s5_tile_inter_block32x64_vert_root_round_trip() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let half = make_inter_cell(
            BlockSize::Block32x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, half);
        splat_cell(&mut grid, mi_cols as usize, 0, 4, half);
        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].2.block_size, BlockSize::Block32x64);
        assert_eq!(leaves[1].2.block_size, BlockSize::Block32x64);
    }

    /// 64x32 inter block (full-tile HORZ at root). Exercises the
    /// inter partition writer's HORZ case at the 64x64 → 64x32 level.
    #[test]
    fn s5_tile_inter_block64x32_horz_root_round_trip() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let half = make_inter_cell(
            BlockSize::Block64x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, half);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, half);
        let leaves = encode_then_replay_inter(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
        );
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].2.block_size, BlockSize::Block64x32);
        assert_eq!(leaves[1].2.block_size, BlockSize::Block64x32);
    }

    // ----- Stage 5: write_tile_inter validation error paths

    /// A half-height bottom SB can use the single visible HORZ leaf.
    #[test]
    fn s5_tile_inter_round_trips_half_height_bottom_sb() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 4u32;
        let mi_cols = 8u32;
        let half = make_inter_cell(
            BlockSize::Block64x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let grid = vec![Some(half); (mi_rows * mi_cols) as usize];
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
        assert_eq!(leaves[0].2.block_size, BlockSize::Block64x32);
    }

    /// `write_tile_inter` rejects tile bounds outside frame.
    #[test]
    fn s5_tile_inter_rejects_invalid_tile_bounds_out_of_frame() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 16, // > mi_rows
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 256];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
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
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, ModesWriterError::InvalidTileBounds));
    }

    /// `write_tile_inter` rejects mismatched prev_mv_grid length.
    #[test]
    fn s5_tile_inter_rejects_invalid_prev_mv_grid_len() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        // prev_mv_grid too short.
        let prev_mv_grid = vec![PrevFrameMvRef::default(); 4];
        let mut buf = vec![0u8; 256];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
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
            Some(&prev_mv_grid),
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, ModesWriterError::InvalidTileBounds));
    }

    /// `write_tile_inter` rejects mismatched mi_grid length.
    #[test]
    fn s5_tile_inter_rejects_invalid_grid_length() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        // Grid too short.
        let grid: Vec<Option<ModeInfo>> = vec![None; 4];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 256];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
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
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, ModesWriterError::InvalidTileBounds));
    }

    /// Count-only and arithmetic traversals gather identical complete counts
    /// even when their probability tables differ.
    #[test]
    fn s5_tile_inter_count_only_matches_emit_across_probability_updates() {
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
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        let mut count_only_counts = FrameCounts::default();
        let mut count_sink = [];
        let mut count_only_bw = BoolWriter::start_count_only(&mut count_sink);
        let mut count_only_coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let mut count_fc = fc.clone();
        count_fc.skip_probs.fill(17);
        for tx_probs in &mut count_fc.coef_probs {
            for plane_probs in tx_probs {
                for ref_probs in plane_probs {
                    for band_probs in ref_probs {
                        for ctx_probs in band_probs {
                            *ctx_probs = [19, 37, 73];
                        }
                    }
                }
            }
        }
        write_tile_inter(
            &mut count_only_bw,
            &count_fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut count_only_coef_ctx,
            &coef_source,
            Some(&mut count_only_counts),
        )
        .expect("count-only traversal");
        assert_eq!(
            count_only_counts, counts,
            "count-only and arithmetic writers must gather identical counts"
        );
        // Single non-skip 64x64 leaf → one entry in partition, skip,
        // intra_inter, single_ref, inter_mode, tx, plus coefficient EOBs.
        let total_partition: u32 = counts.partition.iter().map(|c| c.iter().sum::<u32>()).sum();
        assert_eq!(total_partition, 1, "partition counted once");
        let total_skip: u32 = counts.skip.iter().map(|c| c[0] + c[1]).sum();
        assert_eq!(total_skip, 1);
        let total_intra_inter: u32 = counts.intra_inter.iter().map(|c| c[0] + c[1]).sum();
        assert_eq!(total_intra_inter, 1);
        let total_inter_mode: u32 = counts
            .inter_mode
            .iter()
            .map(|c| c.iter().sum::<u32>())
            .sum();
        assert_eq!(total_inter_mode, 1);
        let total_coef_eob: u32 = counts
            .coef
            .coef
            .iter()
            .flat_map(|tx| tx.iter())
            .flat_map(|plane| plane.iter())
            .flat_map(|reference| reference.iter())
            .flat_map(|band| band.iter())
            .map(|ctx| ctx[crate::entropy::EOB_MODEL_TOKEN])
            .sum();
        assert!(total_coef_eob > 0, "non-skip residuals must count EOBs");

        let mut split_counts = FrameCounts::default();
        let mut retained = crate::entropy::RetainedCoefTokens::default();
        let mut record_pass = CoefTokenPass::record(&mut retained);
        let mut record_sink = [];
        let mut record_bw = BoolWriter::start_count_only(&mut record_sink);
        let mut record_coef_ctx = TileCoefCtx::new(mi_cols as usize);
        write_tile_inter_with_retained(
            &mut record_bw,
            &count_fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut record_coef_ctx,
            &coef_source,
            Some(&mut split_counts),
            Some(&mut record_pass),
        )
        .expect("coefficient-only record traversal");
        assert_eq!(split_counts.coef, counts.coef);
        assert_eq!(split_counts.partition, FrameCounts::default().partition);
        assert_eq!(split_counts.skip, FrameCounts::default().skip);

        let mut full_record_counts = FrameCounts::default();
        let mut full_retained = crate::entropy::RetainedCoefTokens::default();
        let mut full_record_pass = CoefTokenPass::record_full(&mut full_retained);
        let mut full_record_sink = [];
        let mut full_record_bw = BoolWriter::start_count_only(&mut full_record_sink);
        let mut full_record_coef_ctx = TileCoefCtx::new(mi_cols as usize);
        write_tile_inter_with_retained(
            &mut full_record_bw,
            &count_fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut full_record_coef_ctx,
            &coef_source,
            Some(&mut full_record_counts),
            Some(&mut full_record_pass),
        )
        .expect("full count-and-record traversal");
        assert_eq!(full_record_counts, counts);
        assert_eq!(full_retained, retained);

        let mut leaf_counts = FrameCounts::default();
        let mut leaf_retained = crate::entropy::RetainedCoefTokens::default();
        let mut leaf_coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let mut leaf_sink = [];
        let mut leaf_bw = BoolWriter::start_count_only(&mut leaf_sink);
        record_inter_leaf_coefficients(
            &mut leaf_bw,
            &count_fc,
            TxMode::AllowTx32x32,
            &mut leaf_coef_ctx,
            &coef_source,
            0,
            0,
            0,
            BlockSize::Block64x64,
            &cell,
            &mut leaf_counts,
            &mut leaf_retained,
        );
        assert_eq!(leaf_counts.coef, split_counts.coef);
        assert_eq!(leaf_retained, retained);

        let mut replay_counts = FrameCounts::default();
        let mut replay_pass = CoefTokenPass::replay(&retained);
        let mut replay_buf = vec![0u8; 4 * 1024];
        let mut replay_bw = BoolWriter::start_encode(&mut replay_buf);
        let mut replay_coef_ctx = TileCoefCtx::new(mi_cols as usize);
        write_tile_inter_with_retained(
            &mut replay_bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut replay_coef_ctx,
            &coef_source,
            Some(&mut replay_counts),
            Some(&mut replay_pass),
        )
        .expect("retained replay traversal");
        assert_eq!(replay_counts.coef, FrameCounts::default().coef);
        split_counts.add_non_coef(&replay_counts);
        assert_eq!(
            split_counts, counts,
            "coefficient-only record plus final mode counts must equal one full traversal"
        );
    }

    /// ModesWriterError Display impl coverage — each variant.
    #[test]
    fn s5_modes_writer_error_display_missing_cell() {
        let e = ModesWriterError::MissingCell {
            mi_row: 1,
            mi_col: 2,
        };
        let s = format!("{e}");
        assert!(s.contains("missing cell"));
        assert!(s.contains('1') && s.contains('2'));
    }

    #[test]
    fn s5_modes_writer_error_display_unsupported_block_size() {
        let e = ModesWriterError::UnsupportedBlockSize {
            mi_row: 0,
            mi_col: 0,
            block_size: BlockSize::Block8x8,
        };
        let s = format!("{e}");
        assert!(s.contains("block_size"));
    }

    #[test]
    fn s5_modes_writer_error_display_inconsistent_partition_tree() {
        let e = ModesWriterError::InconsistentPartitionTree {
            mi_row: 0,
            mi_col: 0,
            parent_bsize: BlockSize::Block32x32,
            leaf_bsize: BlockSize::Block16x32,
        };
        let s = format!("{e}");
        assert!(s.contains("parent"));
        assert!(s.contains("leaf"));
    }

    #[test]
    fn s5_modes_writer_error_display_unsupported_block_mode() {
        let e = ModesWriterError::UnsupportedBlockMode {
            mi_row: 5,
            mi_col: 6,
        };
        let s = format!("{e}");
        assert!(s.contains("intra"));
    }

    #[test]
    fn s5_modes_writer_error_display_unsupported_non_skip() {
        let e = ModesWriterError::UnsupportedNonSkip {
            mi_row: 0,
            mi_col: 0,
        };
        let s = format!("{e}");
        assert!(s.contains("skip"));
    }

    #[test]
    fn s5_modes_writer_error_display_unsupported_segmentation() {
        let e = ModesWriterError::UnsupportedSegmentation {
            mi_row: 0,
            mi_col: 0,
        };
        let s = format!("{e}");
        assert!(s.contains("segment_id"));
    }

    #[test]
    fn s5_modes_writer_error_display_non_multiple_of_sb() {
        let e = ModesWriterError::NonMultipleOfSb {
            mi_rows: 5,
            mi_cols: 9,
        };
        let s = format!("{e}");
        assert!(s.contains("multiples of 8"));
    }

    #[test]
    fn s5_modes_writer_error_display_invalid_tile_bounds() {
        let e = ModesWriterError::InvalidTileBounds;
        let s = format!("{e}");
        assert!(s.contains("tile bounds"));
    }

    #[test]
    fn s5_modes_writer_error_display_buffer_overflow() {
        let e = ModesWriterError::BufferOverflow;
        let s = format!("{e}");
        assert!(s.contains("overflow"));
    }

    #[test]
    fn s5_modes_writer_error_display_unsupported_intra_in_inter() {
        let e = ModesWriterError::UnsupportedIntraInInter {
            mi_row: 0,
            mi_col: 0,
        };
        let s = format!("{e}");
        assert!(s.contains("intra-in-inter"));
    }

    #[test]
    fn s5_modes_writer_error_display_inter_mode_write() {
        let e = ModesWriterError::InterModeWrite {
            mi_row: 0,
            mi_col: 0,
            source: InterModeWriteError::NotInter,
        };
        let s = format!("{e}");
        assert!(s.contains("inter-mode write"));
    }

    /// `From<ModesWriterError> for BitstreamError` returns
    /// `CompressedHeaderTooLarge` (per the umbrella `From` impl).
    #[test]
    fn s5_modes_writer_error_into_bitstream_error() {
        let e: BitstreamError = ModesWriterError::InvalidTileBounds.into();
        assert!(matches!(e, BitstreamError::CompressedHeaderTooLarge));
    }

    /// `infer_inter_partition` rejects inconsistent BLOCK_8x8 parent
    /// where the leaf is not 4x4/4x8/8x4 (i.e. a non-sub-8 size).
    #[test]
    fn s5_infer_inter_partition_rejects_inconsistent_block8x8_leaf() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // Place a Block16x16 cell at position 0 (the partition walker
        // will expect a Block8x8 or sub-8 leaf for a BLOCK_8x8 parent).
        let bad_cell = make_inter_cell(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        grid[0] = Some(bad_cell);
        // Direct call to `infer_inter_partition` with bsize=Block8x8.
        let err = infer_inter_partition(&grid, mi_cols as usize, 0, 0, BlockSize::Block8x8)
            .expect_err("inconsistent");
        assert!(matches!(
            err,
            ModesWriterError::InconsistentPartitionTree { .. }
        ));
    }

    /// `infer_inter_partition` returns SPLIT for the canonical
    /// Block8x8 case when leaf is Block4x4.
    #[test]
    fn s5_infer_inter_partition_block8x8_split_into_4x4() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = make_inter_cell(
            BlockSize::Block4x4,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        grid[0] = Some(cell);
        let p =
            infer_inter_partition(&grid, mi_cols as usize, 0, 0, BlockSize::Block8x8).expect("ok");
        assert_eq!(p, PartitionType::Split);
    }

    /// SPLIT partition with FrameCounts populated — covers
    /// `counts.as_deref_mut()` paths in the inter partition recursion.
    #[test]
    fn s5_tile_inter_split_accumulates_frame_counts() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = make_inter_cell(
            BlockSize::Block32x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // 4 quadrants × Block32x32 → root partition SPLIT.
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 0, 4, cell);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 4, 4, cell);
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 8 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        // 1 SPLIT at root + 4 NONE at the 32x32 leaves = 5 partition
        // entries.
        let total_partition: u32 = counts.partition.iter().map(|c| c.iter().sum::<u32>()).sum();
        assert_eq!(total_partition, 5);
        // 4 inter leaves → 4 skip entries, 4 inter_mode entries.
        let total_skip: u32 = counts.skip.iter().map(|c| c[0] + c[1]).sum();
        assert_eq!(total_skip, 4);
        let total_inter_mode: u32 = counts
            .inter_mode
            .iter()
            .map(|c| c.iter().sum::<u32>())
            .sum();
        assert_eq!(total_inter_mode, 4);
    }

    /// HORZ partition with FrameCounts populated — covers
    /// `counts.as_deref_mut()` in HORZ partition path.
    #[test]
    fn s5_tile_inter_horz_accumulates_frame_counts() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = make_inter_cell(
            BlockSize::Block64x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, cell);
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        // 1 partition + 2 inter leaves.
        let total_partition: u32 = counts.partition.iter().map(|c| c.iter().sum::<u32>()).sum();
        assert_eq!(total_partition, 1, "1 HORZ partition at root");
        let total_skip: u32 = counts.skip.iter().map(|c| c[0] + c[1]).sum();
        assert_eq!(total_skip, 2);
    }

    /// VERT partition with FrameCounts populated.
    #[test]
    fn s5_tile_inter_vert_accumulates_frame_counts() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = make_inter_cell(
            BlockSize::Block32x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 0, 4, cell);
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        let total_partition: u32 = counts.partition.iter().map(|c| c.iter().sum::<u32>()).sum();
        assert_eq!(total_partition, 1);
    }

    /// `write_tile_intra_keyframe` rejects invalid tile bounds.
    #[test]
    fn s5_tile_intra_keyframe_rejects_invalid_tile_bounds() {
        let fc = FrameContext::new_default();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let grid = make_dc_grid(mi_rows as usize, mi_cols as usize);
        let bad_bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 16, // > mi_rows
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 256];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        let err = write_tile_intra_keyframe(
            &mut bw,
            &grid,
            mi_rows,
            mi_cols,
            bad_bounds,
            &fc,
            TxMode::AllowTx32x32,
            None,
            &mut coef_ctx,
            &coef_source,
        )
        .unwrap_err();
        assert!(matches!(err, ModesWriterError::InvalidTileBounds));
    }

    /// `write_tile_intra_keyframe` rejects mismatched mi_grid length.
    #[test]
    fn s5_tile_intra_keyframe_rejects_invalid_grid_length() {
        let fc = FrameContext::new_default();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let grid: Vec<Option<ModeInfo>> = vec![None; 4];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 256];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        let err = write_tile_intra_keyframe(
            &mut bw,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            &fc,
            TxMode::AllowTx32x32,
            None,
            &mut coef_ctx,
            &coef_source,
        )
        .unwrap_err();
        assert!(matches!(err, ModesWriterError::InvalidTileBounds));
    }

    /// `write_inter_block_leaf` rejects `UnsupportedBlockSize` when
    /// the leaf cell's block_size differs from the partition-derived
    /// subsize. Triggered via an inconsistent grid where the leaf
    /// cell has a smaller bsize than the partition tree expects.
    #[test]
    fn s5_tile_inter_rejects_unsupported_block_size_mismatch() {
        // We need an inter partition tree that derives a subsize via
        // `infer_inter_partition` then dispatches to
        // `write_inter_block_leaf`, where the cell's block_size
        // doesn't match the derived subsize. The cleanest way to
        // trigger this: place a wrong-size cell in a position the
        // tree wouldn't normally infer correctly.
        //
        // Approach: place a Block16x16 cell that splats over (4, 4)
        // region as expected for a 32x32 quadrant; the partition is
        // inferred via the (0,0) cell. Set (0,0) = Block32x32 leaf
        // (NONE partition) but splat only (0,0) with this cell —
        // the rest of (4..) stays None → MissingCell during HORZ/
        // VERT/SPLIT recursion.
        //
        // Actually trigger the simpler `UnsupportedBlockSize`: pass
        // a single 64x64 cell whose stored block_size is Block32x32
        // (mismatched against the NONE-inferred 64x64).
        let mut grid: Vec<Option<ModeInfo>> = vec![None; 64];
        let bad_cell = ModeInfo {
            block_size: BlockSize::Block32x32, // claim 32x32...
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        // ...but splat across the full 64x64 grid (so infer_inter_partition
        // sees Block32x32 at (0,0) → expects HORZ/VERT/SPLIT subsize at
        // BLOCK_64x64 parent). Actually infer_inter_partition will pick
        // a partition matching Block32x32 — that's SPLIT. Then the
        // recursive call at 32x32 finds the same cell, NONE → leaf
        // expects Block32x32 to match. Won't trigger
        // UnsupportedBlockSize.
        //
        // Try mismatched leaf via direct `write_inter_block_leaf` call:
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        grid.fill(Some(bad_cell));
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let mut buf = vec![0u8; 256];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(8);
        // Call write_inter_block_leaf with mismatched leaf_bsize.
        let err = write_inter_block_leaf(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            8,
            8,
            TileBounds {
                mi_row_start: 0,
                mi_row_end: 8,
                mi_col_start: 0,
                mi_col_end: 8,
            },
            0,
            0,
            BlockSize::Block64x64, // leaf_bsize expected
            &bad_cell,             // but cell claims Block32x32
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
            None,
        )
        .expect_err("expected UnsupportedBlockSize");
        assert!(matches!(err, ModesWriterError::UnsupportedBlockSize { .. }));
    }

    /// SPLIT down to 16x16 leaves with FrameCounts populated. Covers
    /// TX size counts for Tx16x16 (line 1910-1913).
    #[test]
    fn s5_tile_inter_block16x16_split_with_counts() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = make_inter_cell(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(2) {
                splat_cell(&mut grid, mi_cols as usize, r, c, cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 16 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        // 16 16x16 leaves → 16 inter_mode counts.
        let total: u32 = counts
            .inter_mode
            .iter()
            .map(|c| c.iter().sum::<u32>())
            .sum();
        assert_eq!(total, 16);
    }

    /// SPLIT down to 8x8 leaves with FrameCounts populated. Covers
    /// TX size counts for Tx8x8 (line 1914-1917).
    #[test]
    fn s5_tile_inter_block8x8_with_counts() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        for r in 0..mi_rows as usize {
            for c in 0..mi_cols as usize {
                splat_cell(&mut grid, mi_cols as usize, r, c, cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 16 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        // 64 8x8 leaves → 64 inter_mode counts.
        let total: u32 = counts
            .inter_mode
            .iter()
            .map(|c| c.iter().sum::<u32>())
            .sum();
        assert_eq!(total, 64);
    }

    /// `infer_partition` returns `InconsistentPartitionTree` when
    /// the leaf size at a non-Block8x8 parent isn't a legal subsize
    /// (lines 676-681).
    #[test]
    fn s5_infer_partition_rejects_inconsistent_keyframe() {
        // A larger rectangular leaf cannot appear below the square
        // parent reached by the recursive tree walk.
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // Place a Block16x32 cell at (0,0). The walker reaches
        // (0,0) at Block16x16 parent and infers... Block16x32 isn't
        // a Block16x16 subsize.
        let cell = ModeInfo {
            block_size: BlockSize::Block16x32,
            tx_size: TxSize::Tx16x16,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        // Fill the rest with valid Block16x16 cells.
        let stamp = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx16x16,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(2) {
                if r < 4 && c < 2 {
                    continue;
                }
                splat_cell(&mut grid, mi_cols as usize, r, c, stamp);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        let result = write_tile_intra_keyframe(
            &mut bw,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            &fc(),
            TxMode::AllowTx32x32,
            None,
            &mut coef_ctx,
            &coef_source,
        );
        // Should reject with either InconsistentPartitionTree (from
        // infer_partition) or UnsupportedBlockSize (from leaf
        // validation). Both are valid outcomes for this malformed
        // grid.
        let err = result.unwrap_err();
        assert!(
            matches!(err, ModesWriterError::InconsistentPartitionTree { .. })
                || matches!(err, ModesWriterError::UnsupportedBlockSize { .. }),
            "actual: {err:?}",
        );
    }

    fn fc() -> FrameContext {
        FrameContext::new_default()
    }

    /// `infer_inter_partition` rejects inconsistent partition tree at
    /// a non-Block8x8 parent (lines 752-757).
    #[test]
    fn s5_infer_inter_partition_rejects_inconsistent_at_non_block8x8() {
        // Block16x32 cell at (0,0) under Block16x16 parent: not a
        // legal subsize. The walker handles 32x32 SPLIT → 16x16
        // children. At the 16x16 parent we hit infer_inter_partition
        // with a Block16x32 cell.
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let bad_cell = make_inter_cell(
            BlockSize::Block16x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, bad_cell);
        let stamp = make_inter_cell(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(2) {
                if r < 4 && c < 2 {
                    continue;
                }
                splat_cell(&mut grid, mi_cols as usize, r, c, stamp);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let err = write_tile_inter(
            &mut bw,
            &fc(),
            &s4c_default_params(),
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, ModesWriterError::InconsistentPartitionTree { .. })
                || matches!(err, ModesWriterError::UnsupportedBlockSize { .. }),
            "actual: {err:?}",
        );
    }

    /// Sub-8 keyframe write: BLOCK_8x8 parent with intra modes
    /// covers the sub-8 plane extent path at line 1182.
    /// We pass an 8x8 cell that's intra DC and let the regular
    /// path handle it; the intra keyframe path doesn't actually
    /// support sub-8 partitioning, but we can drive non-skip
    /// 8x8 keyframe intra to hit the residual write_intra_block_residual
    /// at smaller plane sizes.
    #[test]
    fn s5_intra_keyframe_non_skip_block8x8_grain() {
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        for r in 0..mi_rows as usize {
            for c in 0..mi_cols as usize {
                splat_cell(&mut grid, mi_cols as usize, r, c, cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 64 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        write_tile_intra_keyframe(
            &mut bw,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            &fc(),
            TxMode::AllowTx32x32,
            None,
            &mut coef_ctx,
            &coef_source,
        )
        .expect("ok");
    }

    /// `write_block_intra_keyframe` rejects `seg_id_predicted` (line
    /// 1048).
    #[test]
    fn s5_intra_keyframe_rejects_seg_id_predicted() {
        let fc = FrameContext::new_default();
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut nbr_ctx = NeighbourCtx::new(8);
        let mut bad = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: true,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let _ = &mut bad; // silence unused-mut for future edits
        let grid: Vec<Option<ModeInfo>> = vec![Some(bad); 64];
        let mut buf = vec![0u8; 256];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(8);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        let err = write_block_intra_keyframe(
            &mut bw,
            &fc,
            TxMode::AllowTx32x32,
            None,
            &mut nbr_ctx,
            &bounds,
            &grid,
            8,
            8,
            0,
            0,
            BlockSize::Block64x64,
            &bad,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ModesWriterError::UnsupportedSegmentation { .. }
        ));
    }

    /// `write_block_intra_keyframe` rejects non-zero segment_id when
    /// seg is None (line 1053).
    #[test]
    fn s5_intra_keyframe_rejects_nonzero_seg_id_without_seg_params() {
        let fc = FrameContext::new_default();
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut nbr_ctx = NeighbourCtx::new(8);
        let bad = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 3,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let grid: Vec<Option<ModeInfo>> = vec![Some(bad); 64];
        let mut buf = vec![0u8; 256];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(8);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        let err = write_block_intra_keyframe(
            &mut bw,
            &fc,
            TxMode::AllowTx32x32,
            None,
            &mut nbr_ctx,
            &bounds,
            &grid,
            8,
            8,
            0,
            0,
            BlockSize::Block64x64,
            &bad,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ModesWriterError::UnsupportedSegmentation { .. }
        ));
    }

    /// HORZ partition with second leaf cell missing — should
    /// surface `MissingCell` error (line 1647-1654 path in
    /// write_inter_partition_tree).
    #[test]
    fn s5_tile_inter_horz_partition_missing_bottom_leaf() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // Top half: Block64x32 inter cell.
        let cell = make_inter_cell(
            BlockSize::Block64x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        // Bottom half: deliberately left as None (missing).
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
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
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                ModesWriterError::MissingCell {
                    mi_row: 4,
                    mi_col: 0
                }
            ),
            "actual: {err:?}",
        );
    }

    /// VERT partition with second leaf cell missing.
    #[test]
    fn s5_tile_inter_vert_partition_missing_right_leaf() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = make_inter_cell(
            BlockSize::Block32x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        // Right half missing.
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
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
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                ModesWriterError::MissingCell {
                    mi_row: 0,
                    mi_col: 4
                }
            ),
            "actual: {err:?}",
        );
    }

    /// Keyframe HORZ partition with missing bottom leaf.
    #[test]
    fn s5_tile_intra_keyframe_horz_partition_missing_bottom_leaf() {
        let fc = FrameContext::new_default();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = ModeInfo {
            block_size: BlockSize::Block64x32,
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
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        // Bottom half missing.
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        let err = write_tile_intra_keyframe(
            &mut bw,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            &fc,
            TxMode::AllowTx32x32,
            None,
            &mut coef_ctx,
            &coef_source,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ModesWriterError::MissingCell {
                mi_row: 4,
                mi_col: 0
            }
        ));
    }

    /// Keyframe VERT partition with missing right leaf.
    #[test]
    fn s5_tile_intra_keyframe_vert_partition_missing_right_leaf() {
        let fc = FrameContext::new_default();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = ModeInfo {
            block_size: BlockSize::Block32x64,
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
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        // Right half missing.
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        let err = write_tile_intra_keyframe(
            &mut bw,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            &fc,
            TxMode::AllowTx32x32,
            None,
            &mut coef_ctx,
            &coef_source,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ModesWriterError::MissingCell {
                mi_row: 0,
                mi_col: 4
            }
        ));
    }

    /// `InterModeWriteError` propagates through
    /// `write_inter_block_leaf` as `ModesWriterError::InterModeWrite`.
    /// Trigger via a sub-8 cell with parent mode mismatching bmi[3].
    #[test]
    fn s5_tile_inter_propagates_inter_mode_write_error() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];

        // Build a Block8x8 SB and corrupt sub-8 cell at (0,0) with
        // parent_mode != bmi[3].as_mode.
        let plain = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        for r in 0..mi_rows as usize {
            for c in 0..mi_cols as usize {
                grid[r * mi_cols as usize + c] = Some(plain);
            }
        }
        // Replace (0,0) with a sub-8 cell whose parent mode mismatches.
        let mut bad = ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NewMv, // parent says NEW...
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        for slot in &mut bad.bmi {
            slot.as_mode = InterMode::ZeroMv.idx() as u8; // ...but bmi[3] = ZERO.
        }
        grid[0] = Some(bad);

        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 16 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
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
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                ModesWriterError::InterModeWrite {
                    mi_row: 0,
                    mi_col: 0,
                    ..
                }
            ),
            "actual: {err:?}",
        );
    }

    /// `write_tile_inter` returns `BufferOverflow` when the
    /// underlying BoolWriter runs out of space. Triggered by a
    /// deliberately tiny buffer and a moderately dense grid.
    #[test]
    fn s5_tile_inter_buffer_overflow() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut cell = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [
                Mv {
                    row: 100,
                    col: -100,
                },
                Mv::ZERO,
            ],
            InterpFilter::EightTap,
        );
        cell.skip = false;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        for r in 0..mi_rows as usize {
            for c in 0..mi_cols as usize {
                splat_cell(&mut grid, mi_cols as usize, r, c, cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        // 16-byte buffer is too small to encode an 8x8 grid of inter
        // NEWMV non-skip leaves.
        let mut buf = vec![0u8; 16];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
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
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, ModesWriterError::BufferOverflow));
    }

    /// Sub-8 inter leaf with FrameCounts populated — covers the
    /// Tx4x4 branch of the TX count match (line 1918) since
    /// max_txsize(Block4x4) = Tx4x4.
    #[test]
    fn s5_tile_inter_sub8_with_counts() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // Fill with Block8x8 plain leaves first.
        let plain = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        grid.fill(Some(plain));
        // Place a sub-8 Block4x4 cell at (0, 0). All bmi entries
        // must use ZEROMV (parent mode) for the writer's parent-
        // mode invariant.
        let mut sub8_bmi = [BmiInfo::default(); 4];
        for slot in &mut sub8_bmi {
            slot.as_mode = InterMode::ZeroMv.idx() as u8;
        }
        let sub8_cell = ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: sub8_bmi,
        };
        grid[0] = Some(sub8_cell);
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 16 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        // The Tx4x4 leaf doesn't add to tx_totals, but the rest of
        // the 8x8 leaves use Tx8x8 → tx_totals[1] (Tx8x8 idx 1) > 0.
        let total_tx_8x8: u32 = counts.tx.tx_totals[TxSize::Tx8x8 as usize];
        assert!(total_tx_8x8 > 0, "tx_totals[Tx8x8] should be populated");
    }

    /// Non-skip 64x64 inter leaf with TxMode::AllowTx16x16 →
    /// covers uv_plane_tx_size's `luma_tx < cap` branch (line 2121)
    /// and Block32x16 case in uv_plane_block_size — actually
    /// uv_plane(64x64) = 32x32, so we get the 64x64 case. The cap
    /// scenario: luma_tx = Tx16x16 < cap (Tx32x32) → returns Tx16x16.
    #[test]
    fn s5_tile_inter_non_skip_64x64_with_tx16x16_cap() {
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
        cell.tx_size = TxSize::Tx16x16;
        let grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 64 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx16x16, // cap = Tx16x16
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .expect("ok");
    }

    /// S5 — non-skip SPLIT inter via `encode_then_replay_inter_full`
    /// (exercises the test-side `replay_inter_partition_tree_with_coefs`
    /// HORZ/VERT/SPLIT paths).
    #[test]
    fn s5_round_trip_non_skip_inter_split_via_replay() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let mut cell = make_inter_cell(
            BlockSize::Block32x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.skip = false;
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 0, 4, cell);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 4, 4, cell);

        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let (leaves, _residuals) = encode_then_replay_inter_full(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
            &coef_source,
        );
        assert_eq!(leaves.len(), 4);
        for (_, _, mi) in &leaves {
            assert_eq!(mi.block_size, BlockSize::Block32x32);
        }
    }

    /// S5 — non-skip HORZ inter via the full replay.
    #[test]
    fn s5_round_trip_non_skip_inter_horz_via_replay() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let mut cell = make_inter_cell(
            BlockSize::Block64x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.skip = false;
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 4, 0, cell);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let (leaves, _) = encode_then_replay_inter_full(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
            &coef_source,
        );
        assert_eq!(leaves.len(), 2);
    }

    /// S5 — non-skip VERT inter via the full replay.
    #[test]
    fn s5_round_trip_non_skip_inter_vert_via_replay() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let mut cell = make_inter_cell(
            BlockSize::Block32x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.skip = false;
        splat_cell(&mut grid, mi_cols as usize, 0, 0, cell);
        splat_cell(&mut grid, mi_cols as usize, 0, 4, cell);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let (leaves, _) = encode_then_replay_inter_full(
            &fc,
            &params,
            TxMode::AllowTx32x32,
            mi_rows,
            mi_cols,
            &grid,
            None,
            &coef_source,
        );
        assert_eq!(leaves.len(), 2);
    }

    /// Non-skip 32x16 inter leaves — exercises uv_plane_block_size
    /// `Block32x16 → Block16x8` (line 2108) during residual.
    #[test]
    fn s5_tile_inter_non_skip_32x16_uv_plane() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let mut cell = make_inter_cell(
            BlockSize::Block32x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.skip = false;
        // Fill SB with 32x16 leaves: HORZ in each 32x32 quadrant.
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(4) {
                splat_cell(&mut grid, mi_cols as usize, r, c, cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 64 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .expect("ok");
    }

    /// Intra-in-inter leaf with `TxMode::TxModeSelect` — exercises
    /// the `allow_select && tx_mode == TxModeSelect` branch at line
    /// 1878 (uses mi.tx_size directly).
    #[test]
    fn s5_tile_inter_intra_in_inter_tx_mode_select() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let intra_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx16x16,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let grid: Vec<Option<ModeInfo>> = vec![Some(intra_cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 16 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::TxModeSelect,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .expect("ok");
    }

    /// Intra-in-inter sub-8 leaf at size_group 0 — covers line 1971.
    #[test]
    fn s5_tile_inter_intra_in_inter_sub8_size_group0() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // Plain Block8x8 inter cells everywhere.
        let plain = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        grid.fill(Some(plain));
        // Sub-8 intra cell at (0, 0) — Block4x4 with intra modes.
        let sub8_intra = ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::V,
                uv_mode: IntraMode::H,
            },
            bmi: [BmiInfo::default(); 4],
        };
        grid[0] = Some(sub8_intra);
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 16 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        // size_group 0 (Block4x4) intra leaf with y_mode = V.
        assert_eq!(counts.y_mode[0][IntraMode::V as usize], 1);
    }

    /// `write_tile_intra_keyframe` returns `BufferOverflow` with
    /// a tiny buffer.
    #[test]
    fn s5_tile_intra_keyframe_buffer_overflow() {
        let fc = FrameContext::new_default();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::H,
                uv_mode: IntraMode::V,
            },
            bmi: Default::default(),
        };
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        for r in 0..mi_rows as usize {
            for c in 0..mi_cols as usize {
                splat_cell(&mut grid, mi_cols as usize, r, c, cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 16];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroIntraCoefSource::new();
        let err = write_tile_intra_keyframe(
            &mut bw,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            &fc,
            TxMode::AllowTx32x32,
            None,
            &mut coef_ctx,
            &coef_source,
        )
        .unwrap_err();
        assert!(matches!(err, ModesWriterError::BufferOverflow));
    }

    /// Intra-in-inter leaf at size_group 2 (Block16x16). Covers line
    /// 1973.
    #[test]
    fn s5_tile_inter_intra_in_inter_block16x16_size_group2() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let intra_cell = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx16x16,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::V,
                uv_mode: IntraMode::H,
            },
            bmi: Default::default(),
        };
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(2) {
                splat_cell(&mut grid, mi_cols as usize, r, c, intra_cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 16 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        // 16 intra leaves at size_group 2.
        assert_eq!(counts.y_mode[2][IntraMode::V as usize], 16);
    }

    /// Intra-in-inter leaf at size_group 1 (Block8x8).
    #[test]
    fn s5_tile_inter_intra_in_inter_block8x8_size_group1() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let intra_cell = ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx8x8,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::D45,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        for r in 0..mi_rows as usize {
            for c in 0..mi_cols as usize {
                splat_cell(&mut grid, mi_cols as usize, r, c, intra_cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 32 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        // 64 intra leaves at size_group 1.
        assert_eq!(counts.y_mode[1][IntraMode::D45 as usize], 64);
    }

    /// Non-skip sub-8 (Block4x4) inter leaf — exercises the sub-8
    /// plane extent path in `write_inter_block_residual` (line 2213).
    #[test]
    fn s5_tile_inter_non_skip_sub8_block4x4() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        // Fill SB with plain Block8x8 leaves.
        let plain = make_inter_cell(
            BlockSize::Block8x8,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        grid.fill(Some(plain));
        // Sub-8 cell at (0, 0) — Block4x4 ZEROMV non-skip.
        let mut sub8_bmi = [BmiInfo::default(); 4];
        for slot in &mut sub8_bmi {
            slot.as_mode = InterMode::ZeroMv.idx() as u8;
        }
        let sub8_cell = ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: sub8_bmi,
        };
        grid[0] = Some(sub8_cell);
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 64 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .expect("ok");
    }

    /// Non-skip intra-in-inter leaf at Block16x16 (max_tx = Tx16x16,
    /// so plane_tx != Tx32x32) — exercises `y_mode.tx_type()` for
    /// the Y plane (line 2201).
    #[test]
    fn s5_tile_inter_intra_in_inter_non_skip_16x16_y_tx_type() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let cell = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx16x16,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::H, // → ADST_DCT tx_type for non-32x32
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(2) {
                splat_cell(&mut grid, mi_cols as usize, r, c, cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 64 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .expect("ok");
    }

    /// Non-skip intra-in-inter leaf — exercises `residual_y_mode`
    /// extraction at line 2005, the intra-in-inter residual TX walk
    /// (tx_type via y_mode), and uv_plane_tx_size with cap branches.
    #[test]
    fn s5_tile_inter_intra_in_inter_non_skip() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let intra_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::H,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let grid: Vec<Option<ModeInfo>> = vec![Some(intra_cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 64 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .expect("ok");
    }

    /// Non-skip inter at 16x32 leaf — exercises uv_plane_block_size
    /// `Block16x32 → Block8x16` (line 2107) and the smaller TX paths.
    #[test]
    fn s5_tile_inter_non_skip_16x32_exercise_uv_sizes() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let mut cell = make_inter_cell(
            BlockSize::Block16x32,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.skip = false;
        // Fill the SB with 16x32 leaves.
        for c in (0..mi_cols as usize).step_by(2) {
            splat_cell(&mut grid, mi_cols as usize, 0, c, cell);
            splat_cell(&mut grid, mi_cols as usize, 4, c, cell);
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 64 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .expect("ok");
    }

    /// 16x16 grain non-skip inter leaves with FrameCounts populated.
    /// Covers the residual TX walk (line 2200+) at smaller TX sizes.
    #[test]
    fn s5_tile_inter_non_skip_16x16_with_counts() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let mut grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        let mut cell = make_inter_cell(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        cell.skip = false;
        for r in (0..mi_rows as usize).step_by(2) {
            for c in (0..mi_cols as usize).step_by(2) {
                splat_cell(&mut grid, mi_cols as usize, r, c, cell);
            }
        }
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 64 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        // 16 leaves, 16 skip counts, 16 inter_mode counts.
        let total: u32 = counts
            .inter_mode
            .iter()
            .map(|c| c.iter().sum::<u32>())
            .sum();
        assert_eq!(total, 16);
        // skip=false on each leaf.
        let total_skip: u32 = counts.skip.iter().map(|c| c[0] + c[1]).sum();
        assert_eq!(total_skip, 16);
    }

    /// `write_inter_block_leaf` rejects `seg_id_predicted` set.
    #[test]
    fn s5_inter_block_leaf_rejects_seg_id_predicted() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mut bad_cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        bad_cell.seg_id_predicted = true;
        let grid: Vec<Option<ModeInfo>> = vec![Some(bad_cell); 64];
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        let mut buf = vec![0u8; 256];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(8);
        let err = write_inter_block_leaf(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            8,
            8,
            TileBounds {
                mi_row_start: 0,
                mi_row_end: 8,
                mi_col_start: 0,
                mi_col_end: 8,
            },
            0,
            0,
            BlockSize::Block64x64,
            &bad_cell,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
            None,
        )
        .expect_err("expected UnsupportedSegmentation");
        assert!(matches!(
            err,
            ModesWriterError::UnsupportedSegmentation { .. }
        ));
    }

    /// Intra-in-inter leaf: the writer emits y_mode + uv_mode via
    /// `write_intra_mode`. Exercises the `is_inter == false` branch
    /// of `write_inter_block_leaf`.
    #[test]
    fn s5_tile_inter_intra_in_inter_round_trip() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        // Build an intra-in-inter cell — intra mode info but in an
        // inter frame's writer.
        let intra_cell = ModeInfo {
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
        let grid: Vec<Option<ModeInfo>> = vec![Some(intra_cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            None,
        )
        .expect("intra-in-inter leaf must be accepted");
    }

    /// Intra-in-inter leaf with mode counts populated.
    #[test]
    fn s5_tile_inter_intra_in_inter_accumulates_counts() {
        let fc = FrameContext::new_default();
        let params = s4c_default_params();
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let intra_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::V,
                uv_mode: IntraMode::H,
            },
            bmi: Default::default(),
        };
        let grid: Vec<Option<ModeInfo>> = vec![Some(intra_cell); (mi_rows * mi_cols) as usize];
        let bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut buf = vec![0u8; 4 * 1024];
        let mut counts = FrameCounts::default();
        let mut bw = BoolWriter::start_encode(&mut buf);
        let mut coef_ctx = TileCoefCtx::new(mi_cols as usize);
        let coef_source = crate::bitstream::AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &fc,
            &params,
            TxMode::AllowTx32x32,
            &grid,
            mi_rows,
            mi_cols,
            bounds,
            None,
            None,
            &mut coef_ctx,
            &coef_source,
            Some(&mut counts),
        )
        .expect("ok");
        let total_y_mode: u32 = counts.y_mode.iter().map(|c| c.iter().sum::<u32>()).sum();
        let total_uv_mode: u32 = counts.uv_mode.iter().map(|c| c.iter().sum::<u32>()).sum();
        assert_eq!(total_y_mode, 1, "y_mode counted");
        assert_eq!(total_uv_mode, 1, "uv_mode counted");
        // For a 64x64 leaf, size_group = 3.
        assert_eq!(counts.y_mode[3][IntraMode::V as usize], 1);
    }

    /// Compound + Switchable filter combined — exercises both
    /// branches in the same leaf.
    #[test]
    fn s5_tile_inter_compound_with_switchable_filter_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = s4c_default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        params.interp_filter = InterpFilter::Switchable;
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = make_inter_cell(
            BlockSize::Block64x64,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSharp,
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
        assert_inter_match(&cell, &leaves[0].2, "compound + sharp filter");
    }
}
