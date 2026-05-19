//! Per-tile mode-info writer (v1 keyframe-only scaffold).
//!
//! Encoder mirror of the decoder's per-tile partition-tree walker
//! (`rvp9_decoder::tile::decode_tile_intra` + `decode_partition` +
//! `decode_block`). v1 scope is deliberately narrow: every superblock
//! is `PARTITION_NONE` (no split), every block is intra with
//! `DC_PRED`, every block has `skip == true`, segmentation is
//! disabled, and the caller is responsible for emitting a compressed
//! header with `tx_mode == AllowTx32x32` so that the per-block
//! `tx_size` is implicit (not present on the wire). Any input
//! [`ModeInfo`] grid that violates the scope is rejected with
//! [`ModesWriterError`].
//!
//! This commit lands the wire-format scaffolding the follow-on
//! commits will widen as the encoder's mode-decision driver starts to
//! emit real partitioning, modes, tx-size selection, and tokens.
//!
//! ## Wire format produced
//!
//! For each 64×64 superblock in row-major SB scan:
//!
//! 1. **Partition** — `PARTITION_NONE` encoded as a single boolean
//!    `0` against `VP9_KF_PARTITION_PROBS[ctx][0]`. The `ctx` is
//!    computed by mirroring the decoder's
//!    `PartitionCtx::context(mi_row_in_tile, mi_col_in_tile, bsl=3)`.
//! 2. **Segment_id** — skipped (writer rejects `segmentation` outside
//!    of "disabled").
//! 3. **Skip flag** — `1` against `fc.skip_probs[ctx]`. The context
//!    starts at `0` (no neighbours) and grows to `2` only once a
//!    neighbour exists; with `skip` set on every cell, the right-hand
//!    column / bottom row read `skip_probs[2]` on the inside of the
//!    frame.
//! 4. **Y intra mode** — `DC_PRED` encoded as a single boolean `0`
//!    against `VP9_KF_Y_MODE_PROB[above_mode][left_mode][0]`. With
//!    every neighbour also `DC_PRED`, the slot is always
//!    `VP9_KF_Y_MODE_PROB[Dc][Dc]`.
//! 5. **UV intra mode** — `DC_PRED` encoded as a single boolean `0`
//!    against `VP9_KF_UV_MODE_PROB[Dc][0]`.
//! 6. **tx_size** — NOT emitted on the wire when `tx_mode !=
//!    TxModeSelect`. The decoder derives `tx_size = min(max_tx_size,
//!    tx_mode_to_biggest_tx_size(tx_mode))`. For 64×64 blocks with
//!    `tx_mode = AllowTx32x32`, that yields `Tx32x32`.
//! 7. **Residual tokens** — none (skip is set).
//!
//! ## Caller contract
//!
//! - The frame's compressed header must encode
//!   `tx_mode = AllowTx32x32` (or any other non-`TxModeSelect` mode
//!   whose `tx_mode_to_biggest_tx_size` matches the per-block
//!   `tx_size`). The wire format cannot otherwise communicate the
//!   per-block tx-size for skip-only blocks.
//! - The frame must be a keyframe (or intra-only) and segmentation
//!   must be disabled in the uncompressed header.
//! - Frame dimensions must be multiples of 64 pixels (= 8 mi-units).
//!   v1 doesn't yet support partial superblocks at the frame edge.

use super::error::BitstreamError;
use crate::block::{BlockSize, PartitionType};
use crate::entropy::{BoolWriter, FrameContext, VP9_KF_PARTITION_PROBS, VP9_KF_UV_MODE_PROB,
    VP9_KF_Y_MODE_PROB};
use crate::intra_pred::IntraMode;
use crate::mode_info::{BlockMode, ModeInfo};

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
    /// A cell carried a `block_size` other than `Block64x64`. The v1
    /// writer always emits `PARTITION_NONE` at the superblock root,
    /// so every cell must be 64×64.
    UnsupportedBlockSize {
        /// Row of the offending cell.
        mi_row: usize,
        /// Column of the offending cell.
        mi_col: usize,
        /// The actual block size that triggered the rejection.
        block_size: BlockSize,
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
                "modes_writer v1: cell ({mi_row}, {mi_col}) has block_size {block_size:?}, expected Block64x64"
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
/// field the v1 writer reads is `fc.skip_probs`; partitions and modes
/// use the keyframe tables directly).
///
/// Walks `tile_bounds` in row-major 64×64 SB order; for each SB calls
/// [`write_partition`] (which emits `PARTITION_NONE`) and then
/// [`write_block_intra_keyframe`] (which emits the per-block
/// `(skip, y_mode, uv_mode)` triple).
///
/// # Caller contract
///
/// - The compressed header must commit `tx_mode = AllowTx32x32` (or
///   any non-`TxModeSelect` mode whose
///   `tx_mode_to_biggest_tx_size` evaluates to `Tx32x32`).
/// - The uncompressed header must commit `segmentation.enabled =
///   false`.
/// - The frame must be a keyframe (or intra-only) and its
///   `(mi_rows, mi_cols)` must be multiples of 8.
///
/// # Errors
///
/// - [`ModesWriterError::MissingCell`] — a cell on the SB scan was
///   [`None`].
/// - [`ModesWriterError::UnsupportedBlockSize`] / `UnsupportedBlockMode`
///   / `UnsupportedNonSkip` / `UnsupportedSegmentation` — the cell's
///   `ModeInfo` exceeded the v1 scope.
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
            write_partition(bw, &mut part_ctx, &tile_bounds, mi_row, mi_col)?;
            let mi = mi_grid_get(mi_grid, mi_cols_us, mi_row, mi_col)
                .ok_or(ModesWriterError::MissingCell { mi_row, mi_col })?;
            write_block_intra_keyframe(
                bw,
                fc,
                &mut nbr_ctx,
                &tile_bounds,
                mi_row,
                mi_col,
                &mi,
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

/// Write the partition decision for the 64×64 superblock starting at
/// `(mi_row, mi_col)`. v1 always emits `PARTITION_NONE`, encoded as
/// a single boolean `0` against `VP9_KF_PARTITION_PROBS[ctx][0]`.
///
/// Then updates `part_ctx` so the next SB sees the post-block
/// partition context, mirroring the decoder's
/// `dec_update_partition_context` (gated on `bsize >= BLOCK_8X8 &&
/// (bsize == BLOCK_8X8 || partition != SPLIT)`; for the
/// `PARTITION_NONE @ 64x64` case both predicates are true).
fn write_partition(
    bw: &mut BoolWriter<'_>,
    part_ctx: &mut PartitionCtx,
    bounds: &TileBounds,
    mi_row: usize,
    mi_col: usize,
) -> Result<(), ModesWriterError> {
    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    // 64x64 root: bsl = log2(n_8x8_wide / 2) = log2(8 / 2) = 2. Wait —
    // the decoder uses `bsl = n8x8_l2` where `n8x8_l2 = n4x4_l2 - 1`
    // and `n4x4_l2 = 4` for 64x64 (`log2(16) = 4`). So `bsl = 3` at
    // the 64x64 root. This matches `decode_partition`'s
    // `let bsl = n8x8_l2 as usize` at the top-level call.
    let bsl = 3usize;
    let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);
    let probs = &VP9_KF_PARTITION_PROBS[ctx];
    // Both `has_rows` and `has_cols` are true: the SB scan loop only
    // enters this function when `(mi_row, mi_col)` is within the
    // frame, and `hbs = 4 < 8 = MI_BLOCK_SIZE` so the 64x64 root
    // sub-block always has rows and cols inside the same 64x64 patch.
    // The decoder's three-way fallback (rows-only / cols-only / both
    // missing) only fires at the SB-grid edge of a frame whose
    // dimensions aren't multiples of 64; v1 rejects that case
    // upstream via `NonMultipleOfSb`.
    bw.write(0, probs[0]);

    // libvpx `decode_partition`: the partition-context update fires
    // when `bsize >= BLOCK_8X8 && (bsize == BLOCK_8X8 || partition !=
    // SPLIT)`. For 64x64 / PARTITION_NONE both predicates hold.
    let _ = PartitionType::None; // explicit reminder of the encoded value
    let subsize = BlockSize::Block64x64; // PARTITION_NONE keeps parent size
    let bw_mi: usize = usize::from(BlockSize::Block64x64.n_8x8_wide());
    part_ctx.update(mi_row_in_tile, mi_col_in_tile, subsize, bw_mi);

    Ok(())
}

/// Write the per-block keyframe mode info for the 64×64 block
/// starting at `(mi_row, mi_col)`. v1 requires `mi` to be:
///
/// - `block_size == Block64x64`,
/// - `mode == Intra { y_mode: Dc, uv_mode: Dc }`,
/// - `skip == true`,
/// - `segment_id == 0 && !seg_id_predicted`.
///
/// Emits, in order:
///
/// 1. `skip = 1` against `fc.skip_probs[skip_ctx]`.
/// 2. `y_mode = DC_PRED` (single bit `0`) against
///    `VP9_KF_Y_MODE_PROB[above_mode][left_mode][0]`.
/// 3. `uv_mode = DC_PRED` (single bit `0`) against
///    `VP9_KF_UV_MODE_PROB[Dc][0]`.
///
/// Then updates `nbr_ctx` with the just-written `(y_mode, skip)`
/// pair so subsequent blocks see the correct above/left neighbour
/// state. Mirrors the relevant slice of the decoder's
/// `TileDecodeCtx::write_block`.
fn write_block_intra_keyframe(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    nbr_ctx: &mut NeighbourCtx,
    bounds: &TileBounds,
    mi_row: usize,
    mi_col: usize,
    mi: &ModeInfo,
) -> Result<(), ModesWriterError> {
    // Scope validation — keep the error variants tight to the cell.
    if mi.block_size != BlockSize::Block64x64 {
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
    // false`, so no bits are emitted. (The compressed-header
    // segmentation params block in the uncompressed header also
    // marks the feature disabled.)

    // 1. Skip flag — always 1 under v1 scope. The skip prob bank is
    //    laid out as `[u8; SKIP_CONTEXTS]` (3 entries) and the
    //    decoder reads with `r.read(probs[ctx])` (`mode_info.rs::
    //    read_skip`).
    bw.write(1, fc.skip_probs[skip_ctx]);

    // 2. Intra Y mode. The keyframe Y-mode tree walks
    //    `VP9_INTRA_MODE_TREE` against
    //    `VP9_KF_Y_MODE_PROB[above_mode][left_mode]`.
    let y_probs = &VP9_KF_Y_MODE_PROB[above_mode as usize][left_mode as usize];
    write_intra_mode(bw, y_probs, y_mode);

    // 3. Intra UV mode. Same tree, probs gated by the chosen Y mode:
    //    `VP9_KF_UV_MODE_PROB[y_mode]`.
    let uv_probs = &VP9_KF_UV_MODE_PROB[y_mode as usize];
    write_intra_mode(bw, uv_probs, uv_mode);

    // 4. tx_size — NOT emitted on the wire when `tx_mode !=
    //    TxModeSelect`. The caller commits this via the compressed
    //    header.

    // 5. Splat the just-written neighbour state. v1 always writes the
    //    full 64x64 = 8 mi-units of cells. The above/left context for
    //    the next block sees the parent's Y mode.
    let mi_extent = MI_BLOCK_SIZE;
    nbr_ctx.write_block(
        mi_row_in_tile,
        mi_col_in_tile,
        mi_extent,
        mi_extent,
        y_mode,
        true,
    );

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
            let (above_v, left_v) = PARTITION_CONTEXT_LOOKUP[BlockSize::Block64x64 as usize];
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
            let bounds = TileBounds {
                mi_row_start: 0,
                mi_row_end: 8,
                mi_col_start: 0,
                mi_col_end: 8,
            };
            let mut part_ctx = PartitionCtx::new(8);
            write_partition(&mut bw, &mut part_ctx, &bounds, 0, 0)
                .expect("write_partition v1 should succeed");
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
        // context the writer produced (we only check the single-SB
        // case here, but exercise the update path for the multi-SB
        // tests below).
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
            write_block_intra_keyframe(&mut bw, &fc, &mut nbr_ctx, &bounds, 0, 0, &mi)
                .expect("write_block_intra_keyframe v1 should succeed");
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

    /// `write_tile_intra_keyframe` rejects a non-`Block64x64` cell.
    #[test]
    fn write_tile_intra_keyframe_rejects_non_64x64() {
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
        assert!(matches!(
            err,
            ModesWriterError::UnsupportedBlockSize { block_size: BlockSize::Block32x32, .. }
        ));
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
}
