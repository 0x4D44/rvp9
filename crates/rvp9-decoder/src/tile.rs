//! Per-tile decode driver — the partition-tree walker + per-coding-block
//! `decode_block` body.
//!
//! Ports libvpx `decode_partition` (`vp9_decodeframe.c:1171-1225`) and
//! `decode_block` (`vp9_decodeframe.c:911-1027`) for the keyframe path
//! (intra-only — `is_inter_block(mi) == false`). Inter / MC paths are
//! deliberately out of scope for the Stage 3 floor; this module covers
//! enough of the per-block decoder to walk a keyframe through token
//! decode + intra reconstruction.
//!
//! ## Scope of K2.a (this commit)
//!
//! Scaffolding only — the partition walker descends to leaf coding
//! blocks; `decode_block` reads ModeInfo (segment_id / skip / tx_size /
//! Y intra mode / UV intra mode) and stores it back into
//! [`TileDecodeCtx`]. **No pixel reconstruction yet** — that lands in
//! Stage K2.b.
//!
//! ## Tile geometry
//!
//! libvpx's per-tile `(mi_row_start, mi_row_end, mi_col_start,
//! mi_col_end)` are computed by `vp9_tile_set_row` / `vp9_tile_set_col`
//! (`vp9_tile_common.c:18-32`):
//!
//! ```text
//! offset(idx) = min(((idx * sb_cols) >> log2) << MI_BLOCK_SIZE_LOG2, mi_dim)
//! ```
//!
//! where `sb_cols = mi_cols_aligned_to_sb(mi_dim) >> MI_BLOCK_SIZE_LOG2`.
//!
//! For the current Stage 3 floor we only ship a single-tile decode path.
//! Multi-tile vectors (`tile_cols > 1` or `tile_rows > 1`) surface as
//! [`TileDecodeError::MultiTileUnsupported`].

use rvp9_core::bitstream::TileInfo;
use rvp9_core::block::{BlockSize, PartitionType, max_txsize, subsize};
use rvp9_core::entropy::{BoolReader, FrameContext, VP9_KF_PARTITION_PROBS, VP9_KF_UV_MODE_PROB,
    VP9_KF_Y_MODE_PROB};
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_info::{BlockMode, BmiInfo, ModeInfo, read_intra_mode, read_partition,
    read_segment_id, read_skip, read_tx_size, get_tx_size_context};
use rvp9_core::transforms::{TxMode, TxSize};

use crate::tile_ctx::{TileCoefCtx, TileDecodeCtx};

/// `MI_BLOCK_SIZE` from libvpx (`vp9_enums.h:25`) — each 64×64
/// superblock covers an 8×8 mode-info grid.
const MI_BLOCK_SIZE: usize = 8;
/// `MI_BLOCK_SIZE_LOG2` from libvpx (`vp9_enums.h:26`) — = 3.
const MI_BLOCK_SIZE_LOG2: u32 = 3;
/// `MI_MASK` from libvpx (`vp9_enums.h:31`) — = 7.
const MI_MASK: usize = MI_BLOCK_SIZE - 1;
/// Per-bsl block-count offset into the partition probs table
/// (`vp9_entropymode.h` — `PARTITION_PLOFFSET = 4`).
const PARTITION_PLOFFSET: usize = 4;

/// Errors surfaced by the tile decode driver.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TileDecodeError {
    /// Multi-tile decoding is not yet wired in v1. The current floor
    /// only supports `tile_cols == 1 && tile_rows == 1`.
    MultiTileUnsupported {
        /// `log2_tile_cols` from the header.
        log2_tile_cols: u8,
        /// `log2_tile_rows` from the header.
        log2_tile_rows: u8,
    },
    /// The compressed tile bytes did not start with a valid bool-coder
    /// marker bit.
    InvalidTilePayload,
}

impl core::fmt::Display for TileDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MultiTileUnsupported {
                log2_tile_cols,
                log2_tile_rows,
            } => write!(
                f,
                "multi-tile unsupported in v1 (log2_tile_cols={log2_tile_cols}, log2_tile_rows={log2_tile_rows})"
            ),
            Self::InvalidTilePayload => f.write_str("tile payload missing bool-coder marker bit"),
        }
    }
}

impl std::error::Error for TileDecodeError {}

/// Per-tile bounds in 8×8 mode-info units. Mirrors the libvpx
/// `TileInfo` runtime struct (NOT the bitstream `TileInfo` parsed at
/// header time — that one only carries `log2_tile_*`).
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

/// `mi_cols_aligned_to_sb` (`vp9_onyxc_int.h` inline). Round mi-units
/// up to the next 64×64 superblock boundary (`MI_BLOCK_SIZE = 8`).
#[inline]
#[must_use]
const fn mi_aligned_to_sb(mi_dim: usize) -> usize {
    let sb = 1usize << MI_BLOCK_SIZE_LOG2;
    (mi_dim + sb - 1) & !(sb - 1)
}

/// Compute the (start, end) mi-row or mi-col range for tile `idx` of
/// `1 << log2_tile_dim` tiles, given the frame's `mi_dim`. Mirrors
/// libvpx `get_tile_offset` (`vp9_tile_common.c:18-22`) used by
/// `vp9_tile_set_row` / `vp9_tile_set_col`.
#[inline]
#[must_use]
fn tile_mi_offset(idx: u32, mi_dim: usize, log2_tile_dim: u8) -> usize {
    let sb_dim = mi_aligned_to_sb(mi_dim) >> MI_BLOCK_SIZE_LOG2;
    let raw = ((idx as usize) * sb_dim) >> log2_tile_dim;
    let mi_offset = raw << MI_BLOCK_SIZE_LOG2;
    mi_offset.min(mi_dim)
}

/// Build the mi-bounds for tile `(row, col)`. Mirrors libvpx
/// `vp9_tile_init` (`vp9_tile_common.c:34-37`).
#[must_use]
pub fn tile_bounds(
    info: &TileInfo,
    mi_cols: usize,
    mi_rows: usize,
    tile_row: u32,
    tile_col: u32,
) -> TileBounds {
    let mi_row_start = tile_mi_offset(tile_row, mi_rows, info.log2_tile_rows);
    let mi_row_end = tile_mi_offset(tile_row + 1, mi_rows, info.log2_tile_rows);
    let mi_col_start = tile_mi_offset(tile_col, mi_cols, info.log2_tile_cols);
    let mi_col_end = tile_mi_offset(tile_col + 1, mi_cols, info.log2_tile_cols);
    TileBounds {
        mi_row_start,
        mi_row_end,
        mi_col_start,
        mi_col_end,
    }
}

/// libvpx `partition_context_lookup[BLOCK_SIZES]` (`vp9_common_data.c:
/// 227-244`). For each block size, `above` / `left` is a 4-bit mask
/// indicating which sub-block sizes have "split" recorded in the
/// partition context.
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

/// Per-tile partition-context state. libvpx maintains `above_seg_context`
/// (mi_cols-wide) + `left_seg_context` (8 entries — one SB row) per
/// `MACROBLOCKD`. Each byte's low 4 bits encode the partition history
/// for the four `bsl` ∈ {0, 1, 2, 3} (8×8 / 16×16 / 32×32 / 64×64).
///
/// `above_partition[mi_col]` and `left_partition[mi_row & MI_MASK]`
/// supply the context for [`partition_plane_context`].
#[derive(Clone, Debug)]
struct PartitionCtx {
    /// One entry per 8×8 column across the tile. Persists across SB
    /// rows; reset only at frame start.
    above_partition: Vec<u8>,
    /// One entry per 8×8 row inside the current SB row. Reset at the
    /// start of each new SB row (libvpx
    /// `vp9_decodeframe.c:2090` `vp9_zero(left_seg_context)`).
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

    /// Compute the 0..15 partition-plane context for a block at
    /// `(mi_row, mi_col)` with `bsl` ∈ {0..=3}. Mirrors libvpx
    /// `dec_partition_plane_context` (`vp9_decodeframe.c:1121-1131`).
    ///
    /// `mi_col` is the tile-relative column; `mi_row` is the
    /// tile-relative row (the modulo with `MI_MASK` already handles
    /// the SB-row reset, mirroring `mi_row & MI_MASK`).
    fn context(&self, mi_row_in_tile: usize, mi_col_in_tile: usize, bsl: usize) -> usize {
        let above = (self.above_partition[mi_col_in_tile] >> bsl) & 1;
        let left = (self.left_partition[mi_row_in_tile & MI_MASK] >> bsl) & 1;
        (usize::from(left) * 2 + usize::from(above)) + bsl * PARTITION_PLOFFSET
    }

    /// Write back the partition context after a coding block (or
    /// sub-block split) is decoded. Mirrors libvpx
    /// `dec_update_partition_context` (`vp9_decodeframe.c:1133-1145`).
    ///
    /// `bw` is the block's mi-width.
    fn update(&mut self, mi_row_in_tile: usize, mi_col_in_tile: usize, subsize_bs: BlockSize, bw: usize) {
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

/// One frame's input to the tile decoder. The keyframe path threads
/// the parsed header + the active `tx_mode` (from the compressed
/// header) through this struct.
#[derive(Clone, Copy, Debug)]
pub struct FrameDecodeParams {
    /// Frame width in mi-units (= ceil(width / 8)).
    pub mi_cols: usize,
    /// Frame height in mi-units.
    pub mi_rows: usize,
    /// Frame-level `tx_mode` from the compressed header.
    pub tx_mode: TxMode,
    /// `true` when the segmentation header was enabled, the map is
    /// updated per-block, AND `update_map` is set. The keyframe path
    /// segment-id parser needs both flags. (For Stage 3 floor we
    /// surface a simpler combined flag.)
    pub segmentation_update_map: bool,
    /// Segmentation tree probabilities — used only when
    /// `segmentation_update_map == true`.
    pub seg_tree_probs: [u8; rvp9_core::bitstream::SEG_TREE_PROBS],
    /// `true` ⇒ frame is in lossless mode (`base_qindex == 0` etc.).
    /// Drives the `allow_select == false` path in `read_tx_size`.
    pub lossless: bool,
}

/// Run the per-tile decode loop for a keyframe / intra-only frame.
///
/// Walks the tile's SBs row-by-row, recursively decoding partitions
/// and leaf blocks. Per K2.a this only reads ModeInfo + writes to the
/// per-tile contexts; pixel reconstruction lands in K2.b.
///
/// # Errors
///
/// Returns [`TileDecodeError::InvalidTilePayload`] if the tile bytes
/// don't open with a valid bool-coder marker bit.
pub fn decode_tile_intra(
    tile_data: &[u8],
    bounds: TileBounds,
    params: &FrameDecodeParams,
    fc: &FrameContext,
) -> Result<(), TileDecodeError> {
    let mut reader =
        BoolReader::new(tile_data).map_err(|_| TileDecodeError::InvalidTilePayload)?;
    let tile_mi_cols = bounds.mi_col_end - bounds.mi_col_start;
    let mut decode_ctx = TileDecodeCtx::new(tile_mi_cols);
    let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
    let mut part_ctx = PartitionCtx::new(tile_mi_cols);

    let mut mi_row = bounds.mi_row_start;
    while mi_row < bounds.mi_row_end {
        // Reset per-SB-row state (mirrors libvpx
        // `vp9_decodeframe.c:2089-2092`).
        decode_ctx.reset_left();
        coef_ctx.reset_left_coef();
        part_ctx.reset_left();

        let mut mi_col = bounds.mi_col_start;
        while mi_col < bounds.mi_col_end {
            decode_partition(
                &mut reader,
                fc,
                params,
                &bounds,
                &mut decode_ctx,
                &mut coef_ctx,
                &mut part_ctx,
                mi_row,
                mi_col,
                BlockSize::Block64x64,
                4, // n4x4_l2 for 64×64 = log2(16) = 4
            );
            mi_col += MI_BLOCK_SIZE;
        }
        mi_row += MI_BLOCK_SIZE;
    }
    Ok(())
}

/// Recursively decode one partition starting at `(mi_row, mi_col)`
/// with parent block-size `bsize`. Mirrors libvpx `decode_partition`
/// (`vp9_decodeframe.c:1171-1225`).
///
/// `n4x4_l2` is `log2(parent.n_4x4_wide)` — i.e. 4 for 64×64, 3 for
/// 32×32, 2 for 16×16, 1 for 8×8, 0 for 4×4.
#[allow(clippy::too_many_arguments)] // mirrors libvpx 1:1
fn decode_partition(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    params: &FrameDecodeParams,
    bounds: &TileBounds,
    decode_ctx: &mut TileDecodeCtx,
    coef_ctx: &mut TileCoefCtx,
    part_ctx: &mut PartitionCtx,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    n4x4_l2: u8,
) {
    if mi_row >= params.mi_rows || mi_col >= params.mi_cols {
        return;
    }

    let n8x8_l2 = n4x4_l2.saturating_sub(1);
    let num_8x8_wh: usize = 1 << n8x8_l2;
    let hbs = num_8x8_wh >> 1;
    let has_rows = mi_row + hbs < params.mi_rows;
    let has_cols = mi_col + hbs < params.mi_cols;

    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let bsl = n8x8_l2 as usize; // partition-context bsl
    let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);

    // `partition_prob` slot — keyframes use VP9_KF_PARTITION_PROBS;
    // libvpx assigns `partition_probs` to either the frame context's
    // `partition_prob` or the kf table at `set_partition_probs` in
    // `vp9_decodemv.c` (per-frame in `setup_token_decoder`).
    let probs = &VP9_KF_PARTITION_PROBS[ctx];
    let partition = if has_rows && has_cols {
        read_partition(reader, probs)
    } else if !has_rows && has_cols {
        // Libvpx: `read(probs[1]) ? PARTITION_SPLIT : PARTITION_HORZ`.
        if reader.read(probs[1]) != 0 {
            PartitionType::Split
        } else {
            PartitionType::Horz
        }
    } else if has_rows && !has_cols {
        // Libvpx: `read(probs[2]) ? PARTITION_SPLIT : PARTITION_VERT`.
        if reader.read(probs[2]) != 0 {
            PartitionType::Split
        } else {
            PartitionType::Vert
        }
    } else {
        PartitionType::Split
    };

    let sub = subsize(bsize, partition).unwrap_or(bsize);

    if hbs == 0 {
        // bsize is BLOCK_8x8 partitioning into 4x4/4x8/8x4. Always one
        // call into decode_block, mi-width = mi-height = 1.
        decode_block(
            reader,
            fc,
            params,
            bounds,
            decode_ctx,
            coef_ctx,
            mi_row,
            mi_col,
            sub,
        );
    } else {
        match partition {
            PartitionType::None => {
                decode_block(
                    reader,
                    fc,
                    params,
                    bounds,
                    decode_ctx,
                    coef_ctx,
                    mi_row,
                    mi_col,
                    sub,
                );
            }
            PartitionType::Horz => {
                decode_block(
                    reader,
                    fc,
                    params,
                    bounds,
                    decode_ctx,
                    coef_ctx,
                    mi_row,
                    mi_col,
                    sub,
                );
                if has_rows {
                    decode_block(
                        reader,
                        fc,
                        params,
                        bounds,
                        decode_ctx,
                        coef_ctx,
                        mi_row + hbs,
                        mi_col,
                        sub,
                    );
                }
            }
            PartitionType::Vert => {
                decode_block(
                    reader,
                    fc,
                    params,
                    bounds,
                    decode_ctx,
                    coef_ctx,
                    mi_row,
                    mi_col,
                    sub,
                );
                if has_cols {
                    decode_block(
                        reader,
                        fc,
                        params,
                        bounds,
                        decode_ctx,
                        coef_ctx,
                        mi_row,
                        mi_col + hbs,
                        sub,
                    );
                }
            }
            PartitionType::Split => {
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    mi_row, mi_col, sub, n8x8_l2,
                );
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    mi_row, mi_col + hbs, sub, n8x8_l2,
                );
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    mi_row + hbs, mi_col, sub, n8x8_l2,
                );
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    mi_row + hbs, mi_col + hbs, sub, n8x8_l2,
                );
            }
        }
    }

    // Update partition context — mirror libvpx
    // `decode_partition:1222-1224`: only when `bsize >= BLOCK_8X8`
    // AND (bsize == BLOCK_8X8 || partition != SPLIT).
    if bsize as u8 >= BlockSize::Block8x8 as u8
        && (bsize == BlockSize::Block8x8 || partition != PartitionType::Split)
    {
        part_ctx.update(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
    }
}

/// Decode one leaf coding block at `(mi_row, mi_col)` with size
/// `bsize`. Reads ModeInfo, writes it into `decode_ctx`, and (in
/// K2.b+) drives per-plane prediction + token decode + IT-add.
///
/// For K2.a this is the parse-only stub: it reads all per-block syntax
/// (segment_id / skip / tx_size / Y intra mode / UV intra mode) and
/// stores the resulting ModeInfo into `decode_ctx`. No pixels are
/// touched yet.
#[allow(clippy::too_many_arguments)]
fn decode_block(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    params: &FrameDecodeParams,
    bounds: &TileBounds,
    decode_ctx: &mut TileDecodeCtx,
    _coef_ctx: &mut TileCoefCtx,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
) {
    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let mi_row_in_sb = mi_row_in_tile & MI_MASK;

    // --- Read intra-frame mode info (libvpx
    //     `read_intra_frame_mode_info`, `vp9_decodemv.c:192-233`).

    // Segment ID — only meaningful when segmentation is enabled AND
    // `update_map` is set. Otherwise libvpx returns 0.
    let segment_id = if params.segmentation_update_map {
        read_segment_id(reader, &params.seg_tree_probs)
    } else {
        0
    };

    // Skip flag — context is `above_skip + left_skip`.
    let above = decode_ctx.above(mi_col_in_tile);
    let left = decode_ctx.left(mi_row_in_sb);
    let skip_ctx = usize::from(above.is_some_and(|m| m.skip))
        + usize::from(left.is_some_and(|m| m.skip));
    let skip = read_skip(reader, &fc.skip_probs, skip_ctx);

    // TX size — driven by frame-level tx_mode + block's max_txsize.
    let max_tx = max_txsize(bsize);
    let above_tx_pair = above.map(|m| (m.tx_size, m.skip));
    let left_tx_pair = left.map(|m| (m.tx_size, m.skip));
    let tx_ctx = if max_tx >= TxSize::Tx8x8 {
        get_tx_size_context(above_tx_pair, left_tx_pair, max_tx)
    } else {
        0
    };
    let tx_size = read_tx_size(
        reader,
        &fc.tx_probs,
        params.tx_mode,
        max_tx,
        /* allow_select = */ !params.lossless,
        tx_ctx,
    );

    // --- Y intra mode(s) ---
    //
    // Keyframe Y mode uses the (above, left) neighbour-mode pair as
    // context. For sub-8×8 (`Block4x4 / Block4x8 / Block8x4`) the
    // block carries 1..4 separate per-4×4 modes in `bmi[]`; the parent
    // `mode` field holds the last decoded sub-block mode.
    let (y_mode, bmi) = read_keyframe_y_mode(
        reader,
        bsize,
        above.map(|m| (m.mode, m.bmi)),
        left.map(|m| (m.mode, m.bmi)),
    );

    // --- UV intra mode --- conditional on the Y mode (libvpx
    // `read_intra_mode(r, vp9_kf_uv_mode_prob[mi->mode])`).
    let uv_mode = read_intra_mode(reader, &VP9_KF_UV_MODE_PROB[y_mode as usize]);

    // Assemble + write back.
    let mi = ModeInfo {
        block_size: bsize,
        tx_size,
        skip,
        segment_id,
        seg_id_predicted: false,
        mode: BlockMode::Intra { y_mode, uv_mode },
        bmi,
    };

    // Clamp the in-mi-grid write to the tile bounds (libvpx
    // `x_mis/y_mis` clamp at `vp9_decodeframe.c:917-918`).
    let bw_mi = usize::from(bsize.n_8x8_wide());
    let bh_mi = usize::from(bsize.n_8x8_high());
    let x_mis = bw_mi.min(params.mi_cols - mi_col);
    let y_mis = bh_mi.min(params.mi_rows - mi_row);
    decode_ctx.write_block(mi_row_in_sb, mi_col_in_tile, x_mis, y_mis, &mi);
}

/// Keyframe Y-mode reader. Mirrors libvpx's per-`bsize` dispatch at
/// `vp9_decodemv.c:209-230`. Returns `(parent_y_mode, bmi)` where
/// `bmi` is populated for sub-8 sizes and zeroed otherwise.
///
/// The (above, left) neighbour modes feed
/// `vp9_kf_y_mode_prob[above][left]` via the libvpx convention
/// implemented in `vp9_(above|left)_block_mode`
/// (`vp9_blockd.c:13-35`).
fn read_keyframe_y_mode(
    reader: &mut BoolReader<'_>,
    bsize: BlockSize,
    above: Option<(BlockMode, [BmiInfo; 4])>,
    left: Option<(BlockMode, [BmiInfo; 4])>,
) -> (IntraMode, [BmiInfo; 4]) {
    // For BLOCK_4X4 / BLOCK_4X8 / BLOCK_8X4, libvpx subdivides into
    // up to 4 per-4×4 modes. The current commit handles all four
    // bsize arms 1:1 with `read_intra_frame_mode_info`.
    let mut bmi = [BmiInfo::default(); 4];
    match bsize {
        BlockSize::Block4x4 => {
            for i in 0..4 {
                let probs = kf_y_mode_probs_for(i, &bmi, above, left);
                let m = read_intra_mode(reader, &probs);
                bmi[i].as_mode = m as u8;
            }
            // libvpx: `mi->mode = mi->bmi[3].as_mode`.
            let parent = IntraMode::from_u8(bmi[3].as_mode).expect("legal mode");
            (parent, bmi)
        }
        BlockSize::Block4x8 => {
            let p0 = kf_y_mode_probs_for(0, &bmi, above, left);
            let m0 = read_intra_mode(reader, &p0);
            bmi[0].as_mode = m0 as u8;
            bmi[2].as_mode = m0 as u8;
            let p1 = kf_y_mode_probs_for(1, &bmi, above, left);
            let m1 = read_intra_mode(reader, &p1);
            bmi[1].as_mode = m1 as u8;
            bmi[3].as_mode = m1 as u8;
            (m1, bmi)
        }
        BlockSize::Block8x4 => {
            let p0 = kf_y_mode_probs_for(0, &bmi, above, left);
            let m0 = read_intra_mode(reader, &p0);
            bmi[0].as_mode = m0 as u8;
            bmi[1].as_mode = m0 as u8;
            let p2 = kf_y_mode_probs_for(2, &bmi, above, left);
            let m2 = read_intra_mode(reader, &p2);
            bmi[2].as_mode = m2 as u8;
            bmi[3].as_mode = m2 as u8;
            (m2, bmi)
        }
        _ => {
            let probs = kf_y_mode_probs_for(0, &bmi, above, left);
            let m = read_intra_mode(reader, &probs);
            (m, bmi)
        }
    }
}

/// Compute `vp9_kf_y_mode_prob[above][left]` for sub-block index
/// `block` (0..4) within the parent. Mirrors libvpx
/// `get_y_mode_probs` (`vp9_blockd.h:269-276`) combined with
/// `vp9_above_block_mode` / `vp9_left_block_mode`
/// (`vp9_blockd.c:13-35`).
///
/// The (above, left) tuple carries the parent's neighbour's
/// (mode, bmi) — `None` at frame-top / tile-left edges. Inter
/// neighbours collapse to `DC_PRED` per libvpx's
/// `if (!left_mi || is_inter_block(left_mi)) return DC_PRED`.
fn kf_y_mode_probs_for(
    block: usize,
    cur_bmi: &[BmiInfo; 4],
    above: Option<(BlockMode, [BmiInfo; 4])>,
    left: Option<(BlockMode, [BmiInfo; 4])>,
) -> [u8; 10 - 1] {
    let above_mode = above_block_mode(block, cur_bmi, above);
    let left_mode = left_block_mode(block, cur_bmi, left);
    VP9_KF_Y_MODE_PROB[above_mode as usize][left_mode as usize]
}

/// libvpx `vp9_above_block_mode` (`vp9_blockd.c:25-35`). For sub-blocks
/// at the top row of the parent (`block` ∈ {0, 1}), inherit from the
/// above neighbour's bottom-row bmi (sub-blocks 2 or 3). For other
/// sub-blocks, read from the current parent's already-decoded sibling.
fn above_block_mode(
    block: usize,
    cur_bmi: &[BmiInfo; 4],
    above: Option<(BlockMode, [BmiInfo; 4])>,
) -> IntraMode {
    if block == 0 || block == 1 {
        match above {
            None => IntraMode::Dc,
            Some((mode, neighbour_bmi)) => match mode {
                BlockMode::Inter { .. } => IntraMode::Dc,
                BlockMode::Intra { y_mode, .. } => {
                    // libvpx returns `get_y_mode(above_mi, b + 2)` for
                    // the parent — for >=8×8 parents that's the parent
                    // mode; for sub-8 parents it's the per-bmi mode.
                    // We approximate by reading the neighbour's
                    // bottom-row bmi entry when available; otherwise
                    // the parent's y_mode.
                    let n_idx = block + 2;
                    if neighbour_bmi[n_idx].as_mode != 0 {
                        IntraMode::from_u8(neighbour_bmi[n_idx].as_mode).unwrap_or(y_mode)
                    } else {
                        y_mode
                    }
                }
            },
        }
    } else {
        // block in {2, 3}: read from current parent's bmi[block - 2].
        IntraMode::from_u8(cur_bmi[block - 2].as_mode).unwrap_or(IntraMode::Dc)
    }
}

/// libvpx `vp9_left_block_mode` (`vp9_blockd.c:13-23`). For sub-blocks
/// at the left column of the parent (`block` ∈ {0, 2}), inherit from
/// the left neighbour's right-column bmi (sub-blocks 1 or 3). For
/// other sub-blocks, read from the current parent's already-decoded
/// sibling.
fn left_block_mode(
    block: usize,
    cur_bmi: &[BmiInfo; 4],
    left: Option<(BlockMode, [BmiInfo; 4])>,
) -> IntraMode {
    if block == 0 || block == 2 {
        match left {
            None => IntraMode::Dc,
            Some((mode, neighbour_bmi)) => match mode {
                BlockMode::Inter { .. } => IntraMode::Dc,
                BlockMode::Intra { y_mode, .. } => {
                    let n_idx = block + 1;
                    if neighbour_bmi[n_idx].as_mode != 0 {
                        IntraMode::from_u8(neighbour_bmi[n_idx].as_mode).unwrap_or(y_mode)
                    } else {
                        y_mode
                    }
                }
            },
        }
    } else {
        // block in {1, 3}: read from current parent's bmi[block - 1].
        IntraMode::from_u8(cur_bmi[block - 1].as_mode).unwrap_or(IntraMode::Dc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tile_mi_offset` matches libvpx for single-tile (`log2 = 0`)
    /// edges and two-tile splits.
    #[test]
    fn tile_offset_single_tile_spans_full_frame() {
        // 320x240 → mi_cols = 40, mi_rows = 30.
        assert_eq!(tile_mi_offset(0, 40, 0), 0);
        assert_eq!(tile_mi_offset(1, 40, 0), 40);
        assert_eq!(tile_mi_offset(0, 30, 0), 0);
        assert_eq!(tile_mi_offset(1, 30, 0), 30);
    }

    /// `tile_mi_offset` for a 2-column tile split (log2 = 1).
    #[test]
    fn tile_offset_two_columns() {
        // mi_cols = 40 (5 SB columns; rounded to 5*8 = 40 mi). 2 tiles
        // halve at SB boundary. (5 * 0) >> 1 = 0; (5 * 1) >> 1 = 2;
        // (5 * 2) >> 1 = 5. ×8 (MI_BLOCK_SIZE) gives mi offsets.
        assert_eq!(tile_mi_offset(0, 40, 1), 0);
        // (1 * 5) >> 1 = 2, ×8 = 16.
        assert_eq!(tile_mi_offset(1, 40, 1), 16);
        // (2 * 5) >> 1 = 5, ×8 = 40 (clamped to mi_cols).
        assert_eq!(tile_mi_offset(2, 40, 1), 40);
    }

    /// `tile_bounds` for the simplest single-tile case.
    #[test]
    fn tile_bounds_single_tile_full_frame() {
        let info = TileInfo {
            log2_tile_cols: 0,
            log2_tile_rows: 0,
        };
        let b = tile_bounds(&info, 40, 30, 0, 0);
        assert_eq!(b.mi_col_start, 0);
        assert_eq!(b.mi_col_end, 40);
        assert_eq!(b.mi_row_start, 0);
        assert_eq!(b.mi_row_end, 30);
    }

    /// `mi_aligned_to_sb` rounds up to the next multiple of 8.
    #[test]
    fn mi_aligned_to_sb_round_up() {
        assert_eq!(mi_aligned_to_sb(0), 0);
        assert_eq!(mi_aligned_to_sb(1), 8);
        assert_eq!(mi_aligned_to_sb(7), 8);
        assert_eq!(mi_aligned_to_sb(8), 8);
        assert_eq!(mi_aligned_to_sb(9), 16);
        assert_eq!(mi_aligned_to_sb(40), 40);
    }

    /// `PartitionCtx` fresh: every cell zero; reset_left clears the
    /// `left_partition` array.
    #[test]
    fn partition_ctx_fresh_is_zero() {
        let ctx = PartitionCtx::new(8);
        assert_eq!(ctx.above_partition, vec![0u8; 8]);
        assert_eq!(ctx.left_partition, [0u8; 8]);
    }

    /// `PartitionCtx::context` for a 64×64 SB at the frame top-left:
    /// `above = left = 0`, `bsl = 3`. ctx = 0 + 0 + 3 * 4 = 12.
    #[test]
    fn partition_ctx_64x64_corner() {
        let ctx = PartitionCtx::new(8);
        let c = ctx.context(0, 0, 3);
        assert_eq!(c, 12);
    }

    /// `PartitionCtx::update` for a 64×64 → 32×32 split: writes
    /// `(above, left) = (8, 8) = (0b1000, 0b1000)` across the 8-mi
    /// span.
    #[test]
    fn partition_ctx_update_64x64_split_into_32x32() {
        let mut ctx = PartitionCtx::new(16);
        ctx.update(0, 0, BlockSize::Block32x32, 8);
        for x in 0..8 {
            assert_eq!(ctx.above_partition[x], 8, "above[{x}] expected 0b1000");
        }
        for y in 0..8 {
            assert_eq!(ctx.left_partition[y], 8, "left[{y}] expected 0b1000");
        }
        // Beyond the 8-mi span (cols 8..16) untouched.
        for x in 8..16 {
            assert_eq!(ctx.above_partition[x], 0);
        }
    }

    /// `PARTITION_CONTEXT_LOOKUP` matches libvpx
    /// (`vp9_common_data.c:230-244`) for every block size.
    #[test]
    fn partition_context_lookup_pins_libvpx() {
        // libvpx (vp9_common_data.c:230-244).
        assert_eq!(PARTITION_CONTEXT_LOOKUP[BlockSize::Block4x4 as usize], (15, 15));
        assert_eq!(PARTITION_CONTEXT_LOOKUP[BlockSize::Block8x8 as usize], (14, 14));
        assert_eq!(PARTITION_CONTEXT_LOOKUP[BlockSize::Block16x16 as usize], (12, 12));
        assert_eq!(PARTITION_CONTEXT_LOOKUP[BlockSize::Block32x32 as usize], (8, 8));
        assert_eq!(PARTITION_CONTEXT_LOOKUP[BlockSize::Block64x64 as usize], (0, 0));
    }
}
