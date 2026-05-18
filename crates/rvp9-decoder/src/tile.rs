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
use rvp9_core::frame::Yv12Buffer;
use rvp9_core::inter::mode_info::{InterModeReadParams, get_intra_inter_context,
    read_inter_block_mode_info};
use rvp9_core::inter::mv_ref::BlockBounds;
use rvp9_core::intra_pred::{DcAvailability, EdgeBuffers, IntraMode, build_intra_predictors};
use rvp9_core::mode_info::{BlockMode, BmiInfo, InterMode, ModeInfo, Mv, RefFrame, read_intra_mode,
    read_is_inter, read_partition, read_segment_id, read_skip, read_tx_size, get_tx_size_context};
use rvp9_core::quant::{ac_quant, dc_quant, get_qindex};
use rvp9_core::transforms::{TxMode, TxSize};

use crate::reconstruct::reconstruct_intra_block;
use crate::reconstruct_inter::{InterBlockInputs, reconstruct_inter_block};
use crate::tile_ctx::{TileCoefCtx, TileDecodeCtx, TileNeighbourLookup};

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
    /// Base qindex from the quantization header (`0..=255`).
    pub base_qindex: i32,
    /// Luma DC delta-q from the quantization header (`-15..=15`).
    pub y_dc_delta_q: i32,
    /// Chroma DC delta-q from the quantization header (`-15..=15`).
    pub uv_dc_delta_q: i32,
    /// Chroma AC delta-q from the quantization header (`-15..=15`).
    pub uv_ac_delta_q: i32,
    /// Per-segment alt-Q feature data. `None` for disabled segments.
    /// When `Some(v)`, the resolved qindex for that segment is
    /// `get_qindex(v, seg_abs_delta, base_qindex)`.
    pub seg_alt_q: [Option<i16>; rvp9_core::loopfilter::MAX_SEGMENTS],
    /// `abs_delta` from the segmentation header.
    pub seg_abs_delta: bool,
}

/// Per-plane geometry derived from the frame's chroma subsampling
/// (4:2:0 in v1 — D-01 RESOLVED).
const SUBSAMPLING_X: [usize; 3] = [0, 1, 1];
const SUBSAMPLING_Y: [usize; 3] = [0, 1, 1];

/// Compute the dequant pair `(dc, ac)` for `(plane, segment_id)`.
/// Mirrors libvpx `vp9_init_plane_quantizers`
/// (`vp9_quantize.c:204-245`) for the 8-bit path.
fn plane_dequant(params: &FrameDecodeParams, plane: usize, segment_id: u8) -> (i32, i32) {
    let qindex = if let Some(alt) = params.seg_alt_q.get(segment_id as usize).copied().flatten() {
        get_qindex(alt as i32, params.seg_abs_delta, params.base_qindex)
    } else {
        params.base_qindex
    };
    let (dc_delta, ac_delta) = if plane == 0 {
        (params.y_dc_delta_q, 0)
    } else {
        (params.uv_dc_delta_q, params.uv_ac_delta_q)
    };
    let dc = dc_quant(qindex, dc_delta) as i32;
    let ac = ac_quant(qindex, ac_delta) as i32;
    (dc, ac)
}

/// Run the per-tile decode loop for a keyframe / intra-only frame.
///
/// Walks the tile's SBs row-by-row, recursively decoding partitions
/// and leaf blocks. After K2.b lands, this also drives per-plane
/// intra prediction + token decode + IT-add into `dst_frame`.
///
/// Returns the populated [`TileDecodeCtx`] so callers (e.g. K2.c's
/// loop-filter wireup) can walk the per-block ModeInfo grid after
/// reconstruction completes.
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
    dst_frame: &mut Yv12Buffer,
    mut coef_counts: Option<&mut rvp9_core::entropy::adapt::CoefCountsBank>,
) -> Result<TileDecodeCtx, TileDecodeError> {
    let mut reader =
        BoolReader::new(tile_data).map_err(|_| TileDecodeError::InvalidTilePayload)?;
    let tile_mi_cols = bounds.mi_col_end - bounds.mi_col_start;
    let tile_mi_rows = bounds.mi_row_end - bounds.mi_row_start;
    let mut decode_ctx = TileDecodeCtx::new_with_grid(tile_mi_cols, tile_mi_rows);
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
                dst_frame,
                mi_row,
                mi_col,
                BlockSize::Block64x64,
                4, // n4x4_l2 for 64×64 = log2(16) = 4
                coef_counts.as_deref_mut(),
            );
            mi_col += MI_BLOCK_SIZE;
        }
        mi_row += MI_BLOCK_SIZE;
    }
    Ok(decode_ctx)
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
    dst_frame: &mut Yv12Buffer,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    n4x4_l2: u8,
    mut coef_counts: Option<&mut rvp9_core::entropy::adapt::CoefCountsBank>,
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
            reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
            mi_row, mi_col, sub, coef_counts.as_deref_mut(),
        );
    } else {
        match partition {
            PartitionType::None => {
                decode_block(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                    mi_row, mi_col, sub, coef_counts.as_deref_mut(),
                );
            }
            PartitionType::Horz => {
                decode_block(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                    mi_row, mi_col, sub, coef_counts.as_deref_mut(),
                );
                if has_rows {
                    decode_block(
                        reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                        mi_row + hbs, mi_col, sub, coef_counts.as_deref_mut(),
                    );
                }
            }
            PartitionType::Vert => {
                decode_block(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                    mi_row, mi_col, sub, coef_counts.as_deref_mut(),
                );
                if has_cols {
                    decode_block(
                        reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                        mi_row, mi_col + hbs, sub, coef_counts.as_deref_mut(),
                    );
                }
            }
            PartitionType::Split => {
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row, mi_col, sub, n8x8_l2,
                    coef_counts.as_deref_mut(),
                );
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row, mi_col + hbs, sub, n8x8_l2,
                    coef_counts.as_deref_mut(),
                );
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row + hbs, mi_col, sub, n8x8_l2,
                    coef_counts.as_deref_mut(),
                );
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row + hbs, mi_col + hbs, sub, n8x8_l2,
                    coef_counts,
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
/// `bsize`. Reads the per-block ModeInfo, writes it into `decode_ctx`,
/// then drives per-plane intra prediction + token decode + IT-add
/// into `dst_frame`.
///
/// Mirrors libvpx `decode_block` (`vp9_decodeframe.c:911-1027`) for the
/// intra-only path:
///
/// 1. `vp9_read_mode_info` → reads segment_id / skip / tx_size /
///    intra modes.
/// 2. `dec_reset_skip_context` if `mi.skip`.
/// 3. For each plane × TX position: `predict_and_reconstruct_intra_block`
///    (= K1 edge-builder + H predict + decode + IT-add).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn decode_block(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    params: &FrameDecodeParams,
    bounds: &TileBounds,
    decode_ctx: &mut TileDecodeCtx,
    coef_ctx: &mut TileCoefCtx,
    dst_frame: &mut Yv12Buffer,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    coef_counts: Option<&mut rvp9_core::entropy::adapt::CoefCountsBank>,
) {
    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let mi_row_in_sb = mi_row_in_tile & MI_MASK;

    // Stage 3 stretch M.5 cursor-state instrumentation. Gated by
    // `RVP9_DEBUG_BOOLCURSOR=1`. Logs `(range, value, count, pos)` from
    // the bool-coder before / after each leaf read in `decode_block`
    // for the first Block4x4 of the first SB (`mi_row == 0 && mi_col ==
    // 0 && bsize == Block4x4`). The output is one line per checkpoint
    // — `RVP9_DBG cursor <where> after=(<range>,<value>,<count>,<pos>)`.
    // Companion to the M.4 dq_coefs dump (`RVP9_DEBUG_DQCOEFS=1/2`)
    // landed in `reconstruct.rs` / `detokenize.rs`.
    let trace_cursor = mi_row == bounds.mi_row_start
        && mi_col == bounds.mi_col_start
        && bsize == BlockSize::Block4x4
        && std::env::var("RVP9_DEBUG_BOOLCURSOR").is_ok_and(|v| v == "1");
    macro_rules! dbg_cursor {
        ($reader:expr, $label:literal) => {
            if trace_cursor {
                let (range, value, count, pos) = $reader.cursor_state();
                eprintln!(
                    "RVP9_DBG cursor {} (range={range}, value=0x{value:016x}, count={count}, pos={pos})",
                    $label
                );
            }
        };
    }

    dbg_cursor!(reader, "enter decode_block");

    // --- Read intra-frame mode info (libvpx
    //     `read_intra_frame_mode_info`, `vp9_decodemv.c:192-233`).

    let segment_id = if params.segmentation_update_map {
        read_segment_id(reader, &params.seg_tree_probs)
    } else {
        0
    };
    dbg_cursor!(reader, "after read_segment_id");

    let above = decode_ctx.above(mi_col_in_tile).copied();
    let left = decode_ctx.left(mi_row_in_sb).copied();
    let skip_ctx = usize::from(above.is_some_and(|m| m.skip))
        + usize::from(left.is_some_and(|m| m.skip));
    let skip = read_skip(reader, &fc.skip_probs, skip_ctx);
    if trace_cursor {
        eprintln!("RVP9_DBG decode_block skip_ctx={skip_ctx} skip={skip}");
    }
    dbg_cursor!(reader, "after read_skip");

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
    if trace_cursor {
        eprintln!("RVP9_DBG decode_block max_tx={max_tx:?} tx_ctx={tx_ctx} tx_size={tx_size:?}");
    }
    dbg_cursor!(reader, "after read_tx_size");

    let (y_mode, bmi) = read_keyframe_y_mode_traced(
        reader,
        bsize,
        above.map(|m| (m.block_size, m.mode, m.bmi)),
        left.map(|m| (m.block_size, m.mode, m.bmi)),
        trace_cursor,
    );
    if trace_cursor {
        eprintln!(
            "RVP9_DBG decode_block y_mode={y_mode:?} bmi=[{:?},{:?},{:?},{:?}]",
            bmi[0].as_mode, bmi[1].as_mode, bmi[2].as_mode, bmi[3].as_mode
        );
    }
    dbg_cursor!(reader, "after read_keyframe_y_mode");

    let uv_mode = read_intra_mode(reader, &VP9_KF_UV_MODE_PROB[y_mode as usize]);
    if trace_cursor {
        eprintln!("RVP9_DBG decode_block uv_mode={uv_mode:?}");
    }
    dbg_cursor!(reader, "after read_intra_mode(uv_mode)");

    let mi = ModeInfo {
        block_size: bsize,
        tx_size,
        skip,
        segment_id,
        seg_id_predicted: false,
        mode: BlockMode::Intra { y_mode, uv_mode },
        bmi,
    };

    // Write the ModeInfo into the neighbour grid before we descend into
    // per-plane reconstruction (mirrors libvpx's `set_offsets` storing
    // the parent `mi` into the grid via x_mis/y_mis).
    let bw_mi = usize::from(bsize.n_8x8_wide());
    let bh_mi = usize::from(bsize.n_8x8_high());
    let x_mis = bw_mi.min(params.mi_cols - mi_col);
    let y_mis = bh_mi.min(params.mi_rows - mi_row);
    decode_ctx.write_block(mi_row_in_tile, mi_row_in_sb, mi_col_in_tile, x_mis, y_mis, &mi);

    // --- Reconstruction ----------------------------------------------
    //
    // Snapshot inbound neighbour flags BEFORE write_block: libvpx checks
    // `xd->above_mi != NULL` / `xd->left_mi != NULL` based on
    // frame/tile-edge semantics, not the just-written self-cell.
    let have_above_block = above.is_some();
    let have_left_block = left.is_some();

    let dequant_y = plane_dequant(params, 0, segment_id);
    let dequant_u = plane_dequant(params, 1, segment_id);
    let dequant_v = plane_dequant(params, 2, segment_id);

    reconstruct_intra_planes(
        reader,
        fc,
        coef_ctx,
        dst_frame,
        &mi,
        y_mode,
        uv_mode,
        &bmi,
        mi_row,
        mi_col,
        mi_row_in_tile,
        mi_col_in_tile,
        mi_row_in_sb,
        have_above_block,
        have_left_block,
        dequant_y,
        dequant_u,
        dequant_v,
        params.lossless,
        coef_counts,
    );
}

/// libvpx `dec_reset_skip_context` (`vp9_decodeframe.c:783-790`):
/// zero the per-plane above + left coef-context cells covering the
/// coding block when the block was marked `skip` (no residual). The
/// span is the plane's `n4_w × n4_h` 4×4 cells.
fn reset_skip_context(
    coef_ctx: &mut TileCoefCtx,
    mi: &ModeInfo,
    mi_row_in_tile: usize,
    mi_col_in_tile: usize,
) {
    let mi_row_in_sb = mi_row_in_tile & MI_MASK;
    for plane in 0..3 {
        let ss_x = SUBSAMPLING_X[plane];
        let ss_y = SUBSAMPLING_Y[plane];
        let plane_bsize = if plane == 0 {
            mi.block_size
        } else {
            uv_plane_block_size(mi.block_size)
        };
        let n4_w = plane_bsize.n_4x4_wide() as usize;
        let n4_h = plane_bsize.n_4x4_high() as usize;
        let x4_base = (mi_col_in_tile * 2) >> ss_x;
        let y4_base = (mi_row_in_sb * 2) >> ss_y;
        coef_ctx.set_coef_flags(plane, x4_base, y4_base, n4_w, n4_h, false);
    }
}

/// Compute the chroma plane's block size for the given luma `bsize`,
/// assuming 4:2:0 (D-01 RESOLVED). Mirrors libvpx's
/// `ss_size_lookup[bsize][1][1]`. For sub-8 luma blocks, libvpx pins
/// the chroma plane to `BLOCK_8X8` (the chroma can't subdivide below
/// 4×4 in 4:2:0).
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

/// libvpx `get_uv_tx_size` (`vp9_blockd.h:262-267`): chroma TX size is
/// `min(mi->tx_size, max_txsize_lookup[plane_bsize])`.
fn uv_plane_tx_size(bsize: BlockSize, luma_tx: TxSize) -> TxSize {
    let cap = max_txsize(uv_plane_block_size(bsize));
    if (luma_tx as u8) < (cap as u8) {
        luma_tx
    } else {
        cap
    }
}

/// Per-plane intra reconstruction loop, shared between keyframe
/// [`decode_block`] (true intra frame) and the intra-in-inter arm of
/// [`decode_inter_block`] (D.4).
///
/// Mirrors the inner per-plane / per-TX iteration of libvpx's
/// `predict_and_reconstruct_intra_block` plus the surrounding
/// `dec_reset_skip_context` (`vp9_decodeframe.c:235-294, 783-790`).
/// The mode-info reader before this point differs between keyframe
/// (uses `vp9_kf_y_mode_prob[above][left]`) and inter-frame intra
/// (uses `fc.y_mode_prob[size_group]`), so the caller pre-resolves
/// `(y_mode, uv_mode, bmi, tx_size, skip)` and passes them in.
///
/// Inputs:
/// - `(have_above_block, have_left_block)` snapshot the inbound
///   neighbour-exists flags BEFORE the caller's `write_block` call,
///   matching libvpx's `xd->above_mi != NULL` / `xd->left_mi != NULL`.
/// - `(dequant_y, dequant_u, dequant_v)` are the per-plane
///   `(dc, ac)` dequant pairs from `plane_dequant` (already includes
///   segmentation-aware qindex resolution).
/// - `frame_lossless` propagates the frame-level `lossless` flag —
///   `reconstruct_intra_block` will be called with
///   `frame_lossless && plane_tx == TxSize::Tx4x4` to match libvpx's
///   `inverse_transform_block_intra:280` WHT-vs-IDCT switch.
#[allow(clippy::too_many_arguments)] // mirrors the inlined libvpx loop
fn reconstruct_intra_planes(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    coef_ctx: &mut TileCoefCtx,
    dst_frame: &mut Yv12Buffer,
    mi: &ModeInfo,
    y_mode: IntraMode,
    uv_mode: IntraMode,
    bmi: &[BmiInfo; 4],
    mi_row: usize,
    mi_col: usize,
    mi_row_in_tile: usize,
    mi_col_in_tile: usize,
    mi_row_in_sb: usize,
    have_above_block: bool,
    have_left_block: bool,
    dequant_y: (i32, i32),
    dequant_u: (i32, i32),
    dequant_v: (i32, i32),
    frame_lossless: bool,
    mut coef_counts: Option<&mut rvp9_core::entropy::adapt::CoefCountsBank>,
) {
    let bsize = mi.block_size;
    let tx_size = mi.tx_size;

    if mi.skip {
        // libvpx `dec_reset_skip_context` — zero the per-plane above/
        // left coef-context cells covering the block.
        reset_skip_context(coef_ctx, mi, mi_row_in_tile, mi_col_in_tile);
    }

    let mut edges = EdgeBuffers::new();
    for plane in 0..3 {
        let ss_x = SUBSAMPLING_X[plane];
        let ss_y = SUBSAMPLING_Y[plane];

        // libvpx `get_uv_tx_size` — chroma TX size = min(luma_tx, max_txsize(plane_bsize)).
        let plane_tx = if plane == 0 {
            tx_size
        } else {
            uv_plane_tx_size(bsize, tx_size)
        };
        let plane_bsize = if plane == 0 {
            bsize
        } else {
            uv_plane_block_size(bsize)
        };
        // libvpx pd->n4_w / n4_h: sub-8 coding-block path uses (2 >> ss)
        // regardless of subsize; ≥ 8×8 uses the plane_bsize 4×4 extent.
        let (plane_bw_4x4, plane_bh_4x4) = if bsize < BlockSize::Block8x8 {
            (2 >> ss_x, 2 >> ss_y)
        } else {
            (
                plane_bsize.n_4x4_wide() as usize,
                plane_bsize.n_4x4_high() as usize,
            )
        };
        let tx_step_4x4 = 1usize << (plane_tx as usize);

        let plane_px_x = (mi_col * 8) >> ss_x;
        let plane_px_y = (mi_row * 8) >> ss_y;

        let plane_x4_base = (mi_col_in_tile * 2) >> ss_x;
        let plane_y4_in_sb_base = (mi_row_in_sb * 2) >> ss_y;

        let mut plane_view = match plane {
            0 => dst_frame.y_plane_mut(),
            1 => dst_frame.u_plane_mut(),
            _ => dst_frame.v_plane_mut(),
        };
        let plane_stride = plane_view.stride();
        let plane_width_px = plane_view.width();
        let plane_height_px = plane_view.height();
        let plane_data: &mut [u8] = plane_view.data_mut();

        let plane_dequant_pair = match plane {
            0 => dequant_y,
            1 => dequant_u,
            _ => dequant_v,
        };
        let plane_intra_mode = if plane == 0 { y_mode } else { uv_mode };

        let mut loff = 0usize;
        while loff < plane_bh_4x4 {
            let mut aoff = 0usize;
            while aoff < plane_bw_4x4 {
                let tx_px_x = plane_px_x + aoff * 4;
                let tx_px_y = plane_px_y + loff * 4;

                if tx_px_x >= plane_width_px || tx_px_y >= plane_height_px {
                    aoff += tx_step_4x4;
                    continue;
                }

                let bw_in_4x4 = plane_bw_4x4;
                let have_top = loff > 0 || have_above_block;
                let have_left = aoff > 0 || have_left_block;
                let have_right = (aoff + tx_step_4x4) < bw_in_4x4;

                let mode_for_this_tx = if plane == 0 && bsize < BlockSize::Block8x8 {
                    let bi = (loff << 1) | aoff;
                    IntraMode::from_u8(bmi[bi].as_mode).unwrap_or(plane_intra_mode)
                } else {
                    plane_intra_mode
                };

                build_intra_predictors(
                    plane_data,
                    plane_stride,
                    tx_px_x,
                    tx_px_y,
                    plane_width_px,
                    plane_height_px,
                    plane_tx,
                    mode_for_this_tx,
                    have_top,
                    have_left,
                    have_right,
                    &mut edges,
                );

                let pred_ctx = edges.pred_ctx(plane_tx.side());

                // Stage 3 stretch M.13 — positional intra-pred trace,
                // filtered to (plane, tx_px_x, tx_px_y) range. Used to
                // localize the bottom-right cascade post-M.12 on the
                // 32×32 vector frame 1. Filter format:
                //   RVP9_TRACE_INTRA_AT="plane,x,y" → exact-pixel match
                //   RVP9_TRACE_INTRA_AT="plane,x..,y.."→ "at-or-after" probe
                //   (unset → no trace)
                if let Ok(filter) = std::env::var("RVP9_TRACE_INTRA_AT") {
                    let parts: Vec<&str> = filter.split(',').collect();
                    if parts.len() == 3 {
                        let p_ok = parts[0].parse::<usize>().ok() == Some(plane);
                        fn at_or_after(spec: &str, v: usize) -> bool {
                            if let Some(stripped) = spec.strip_suffix("..") {
                                stripped.parse::<usize>().map(|t| v >= t).unwrap_or(false)
                            } else {
                                spec.parse::<usize>().ok() == Some(v)
                            }
                        }
                        let x_ok = at_or_after(parts[1], tx_px_x);
                        let y_ok = at_or_after(parts[2], tx_px_y);
                        if p_ok && x_ok && y_ok {
                            let bs = plane_tx.side();
                            eprintln!(
                                "RVP9_TRACE_INTRA_AT plane={plane} tx_px=({tx_px_x},{tx_px_y}) bsize={:?} tx_size={:?} mode={mode_for_this_tx:?} have_top={have_top} have_left={have_left} have_right={have_right} aoff={aoff} loff={loff} bw_4x4={bw_in_4x4} above_left={} above={:?} left={:?}",
                                bsize,
                                plane_tx,
                                pred_ctx.above_left,
                                &pred_ctx.above[..2 * bs],
                                &pred_ctx.left[..bs],
                            );
                        }
                    }
                }
                let dst_offset = tx_px_y * plane_stride + tx_px_x;
                let dst_slice = &mut plane_data[dst_offset..];

                // libvpx `dc_pred[left_available][up_available]` dispatch
                // (`vp9_reconintra.c:398`). For DC mode the kernel choice
                // (dc / dc_top / dc_left / dc_128) depends on which
                // neighbours are real; mixing the 129-fill into the
                // average shifts the DC value by ~1 per missing edge,
                // which propagates as a per-block constant cascade
                // through every downstream intra-pred read (M.12).
                let dc_avail = DcAvailability {
                    above_available: have_top,
                    left_available: have_left,
                };

                if mi.skip {
                    rvp9_core::intra_pred::intra_predict(
                        mode_for_this_tx,
                        dst_slice,
                        plane_stride,
                        plane_tx.side(),
                        pred_ctx,
                        dc_avail,
                    );
                } else {
                    let seg_eob = plane_tx.n_coeffs();
                    let _eob = reconstruct_intra_block(
                        reader,
                        fc,
                        coef_counts.as_deref_mut(),
                        coef_ctx,
                        plane,
                        plane_x4_base + aoff,
                        plane_y4_in_sb_base + loff,
                        plane_tx,
                        mode_for_this_tx,
                        pred_ctx,
                        dc_avail,
                        dst_slice,
                        plane_stride,
                        seg_eob,
                        plane_dequant_pair,
                        frame_lossless && plane_tx == TxSize::Tx4x4,
                    );
                }

                aoff += tx_step_4x4;
            }
            loff += tx_step_4x4;
        }
    }
}


/// Keyframe Y-mode reader. Mirrors libvpx's per-`bsize` dispatch at
/// `vp9_decodemv.c:209-230`. Returns `(parent_y_mode, bmi)` where
/// `bmi` is populated for sub-8 sizes and zeroed otherwise.
///
/// The (above, left) neighbour modes feed
/// `vp9_kf_y_mode_prob[above][left]` via the libvpx convention
/// implemented in `vp9_(above|left)_block_mode`
/// (`vp9_blockd.c:13-35`).
#[allow(dead_code)] // retained for tests; live callers use `_traced`
fn read_keyframe_y_mode(
    reader: &mut BoolReader<'_>,
    bsize: BlockSize,
    above: Option<(BlockSize, BlockMode, [BmiInfo; 4])>,
    left: Option<(BlockSize, BlockMode, [BmiInfo; 4])>,
) -> (IntraMode, [BmiInfo; 4]) {
    read_keyframe_y_mode_traced(reader, bsize, above, left, false)
}

/// As [`read_keyframe_y_mode`], but with optional per-bmi instrumentation
/// for the Stage 3 stretch M.5 cursor-state trace. When `trace == true`
/// every bmi-mode read is logged with the resolved `(above_mode,
/// left_mode)` neighbour-mode pair, the 9-entry probs slice indexed by
/// that pair, and the decoded mode. Callers gate `trace` on the
/// per-block `RVP9_DEBUG_BOOLCURSOR` check.
fn read_keyframe_y_mode_traced(
    reader: &mut BoolReader<'_>,
    bsize: BlockSize,
    above: Option<(BlockSize, BlockMode, [BmiInfo; 4])>,
    left: Option<(BlockSize, BlockMode, [BmiInfo; 4])>,
    trace: bool,
) -> (IntraMode, [BmiInfo; 4]) {
    // For BLOCK_4X4 / BLOCK_4X8 / BLOCK_8X4, libvpx subdivides into
    // up to 4 per-4×4 modes. The current commit handles all four
    // bsize arms 1:1 with `read_intra_frame_mode_info`.
    let mut bmi = [BmiInfo::default(); 4];
    let log_bmi = |i: usize, am: IntraMode, lm: IntraMode, probs: &[u8; 9],
                   m: IntraMode, before: (u32, u64, i32, usize),
                   after: (u32, u64, i32, usize)| {
        eprintln!(
            "RVP9_DBG read_intra_mode bmi[{i}] above={am:?} left={lm:?} \
             probs={probs:?} -> {m:?} \
             before=(r={},v=0x{:016x},c={},p={}) \
             after=(r={},v=0x{:016x},c={},p={})",
            before.0, before.1, before.2, before.3,
            after.0, after.1, after.2, after.3,
        );
    };
    match bsize {
        BlockSize::Block4x4 => {
            for i in 0..4 {
                let above_m = above_block_mode(i, &bmi, above);
                let left_m = left_block_mode(i, &bmi, left);
                let probs = VP9_KF_Y_MODE_PROB[above_m as usize][left_m as usize];
                let before = reader.cursor_state();
                let m = read_intra_mode(reader, &probs);
                let after = reader.cursor_state();
                if trace {
                    log_bmi(i, above_m, left_m, &probs, m, before, after);
                }
                bmi[i].as_mode = m as u8;
            }
            // libvpx: `mi->mode = mi->bmi[3].as_mode`.
            let parent = IntraMode::from_u8(bmi[3].as_mode).expect("legal mode");
            (parent, bmi)
        }
        BlockSize::Block4x8 => {
            let above0 = above_block_mode(0, &bmi, above);
            let left0 = left_block_mode(0, &bmi, left);
            let p0 = VP9_KF_Y_MODE_PROB[above0 as usize][left0 as usize];
            let before0 = reader.cursor_state();
            let m0 = read_intra_mode(reader, &p0);
            let after0 = reader.cursor_state();
            if trace { log_bmi(0, above0, left0, &p0, m0, before0, after0); }
            bmi[0].as_mode = m0 as u8;
            bmi[2].as_mode = m0 as u8;
            let above1 = above_block_mode(1, &bmi, above);
            let left1 = left_block_mode(1, &bmi, left);
            let p1 = VP9_KF_Y_MODE_PROB[above1 as usize][left1 as usize];
            let before1 = reader.cursor_state();
            let m1 = read_intra_mode(reader, &p1);
            let after1 = reader.cursor_state();
            if trace { log_bmi(1, above1, left1, &p1, m1, before1, after1); }
            bmi[1].as_mode = m1 as u8;
            bmi[3].as_mode = m1 as u8;
            (m1, bmi)
        }
        BlockSize::Block8x4 => {
            let above0 = above_block_mode(0, &bmi, above);
            let left0 = left_block_mode(0, &bmi, left);
            let p0 = VP9_KF_Y_MODE_PROB[above0 as usize][left0 as usize];
            let before0 = reader.cursor_state();
            let m0 = read_intra_mode(reader, &p0);
            let after0 = reader.cursor_state();
            if trace { log_bmi(0, above0, left0, &p0, m0, before0, after0); }
            bmi[0].as_mode = m0 as u8;
            bmi[1].as_mode = m0 as u8;
            let above2 = above_block_mode(2, &bmi, above);
            let left2 = left_block_mode(2, &bmi, left);
            let p2 = VP9_KF_Y_MODE_PROB[above2 as usize][left2 as usize];
            let before2 = reader.cursor_state();
            let m2 = read_intra_mode(reader, &p2);
            let after2 = reader.cursor_state();
            if trace { log_bmi(2, above2, left2, &p2, m2, before2, after2); }
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
    above: Option<(BlockSize, BlockMode, [BmiInfo; 4])>,
    left: Option<(BlockSize, BlockMode, [BmiInfo; 4])>,
) -> [u8; 10 - 1] {
    let above_mode = above_block_mode(block, cur_bmi, above);
    let left_mode = left_block_mode(block, cur_bmi, left);
    VP9_KF_Y_MODE_PROB[above_mode as usize][left_mode as usize]
}

/// libvpx `get_y_mode(mi, block)` (`vp9_blockd.h:99-101`). For sub-8
/// neighbours (`sb_type < BLOCK_8X8`) returns the per-bmi mode; for
/// `>= 8×8` neighbours returns the parent's `y_mode`. The branch is
/// on the neighbour's block size, NOT on whether the bmi byte is
/// non-zero — `DC_PRED` is encoded as `0` and is a legitimate per-bmi
/// mode, so a `bmi[b].as_mode != 0` "is-populated" check would
/// misclassify legit Dc bmi entries as missing and substitute the
/// parent y_mode, corrupting the `[above][left]` probs row downstream.
#[inline]
fn neighbour_get_y_mode(
    block_size: BlockSize,
    y_mode: IntraMode,
    bmi: &[BmiInfo; 4],
    block: usize,
) -> IntraMode {
    match block_size {
        BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4 => {
            IntraMode::from_u8(bmi[block].as_mode).unwrap_or(y_mode)
        }
        _ => y_mode,
    }
}

/// libvpx `vp9_above_block_mode` (`vp9_blockd.c:25-35`). For sub-blocks
/// at the top row of the parent (`block` ∈ {0, 1}), inherit from the
/// above neighbour via `get_y_mode(above_mi, b + 2)`. For other
/// sub-blocks, read from the current parent's already-decoded sibling.
fn above_block_mode(
    block: usize,
    cur_bmi: &[BmiInfo; 4],
    above: Option<(BlockSize, BlockMode, [BmiInfo; 4])>,
) -> IntraMode {
    if block == 0 || block == 1 {
        match above {
            None => IntraMode::Dc,
            Some((n_bsize, mode, neighbour_bmi)) => match mode {
                BlockMode::Inter { .. } => IntraMode::Dc,
                BlockMode::Intra { y_mode, .. } => {
                    neighbour_get_y_mode(n_bsize, y_mode, &neighbour_bmi, block + 2)
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
/// the left neighbour via `get_y_mode(left_mi, b + 1)`. For other
/// sub-blocks, read from the current parent's already-decoded sibling.
fn left_block_mode(
    block: usize,
    cur_bmi: &[BmiInfo; 4],
    left: Option<(BlockSize, BlockMode, [BmiInfo; 4])>,
) -> IntraMode {
    if block == 0 || block == 2 {
        match left {
            None => IntraMode::Dc,
            Some((n_bsize, mode, neighbour_bmi)) => match mode {
                BlockMode::Inter { .. } => IntraMode::Dc,
                BlockMode::Intra { y_mode, .. } => {
                    neighbour_get_y_mode(n_bsize, y_mode, &neighbour_bmi, block + 1)
                }
            },
        }
    } else {
        // block in {1, 3}: read from current parent's bmi[block - 1].
        IntraMode::from_u8(cur_bmi[block - 1].as_mode).unwrap_or(IntraMode::Dc)
    }
}

// =========================================================================
// Inter-frame driver (Phase D.3)
// =========================================================================
//
// Mirrors the inter arm of libvpx `decode_tile` / `decode_partition` /
// `decode_block` (`vp9_decodeframe.c:911-1027`). Inter frames differ from
// the keyframe path in three ways:
//
// 1. Partition probabilities come from the per-frame
//    `fc.partition_prob[ctx]` (3 entries) instead of the
//    [`VP9_KF_PARTITION_PROBS`] table.
// 2. The per-block ModeInfo header sequence is `read_inter_frame_mode_info`
//    (`vp9_decodemv.c:788-806`): seg_id → skip → is_inter → tx_size → then
//    either `read_intra_block_mode_info` (non-key intra) or
//    `read_inter_block_mode_info` (which D.1 already ports).
// 3. The reconstruction path is `dec_build_inter_predictors_sb` + the
//    per-plane residual decode — D.2 already ports that as
//    [`decode_and_reconstruct_inter_block`].

/// Per-frame inter-decode parameters. Mirrors the shape of
/// [`FrameDecodeParams`] but carries the inter-only fields the
/// per-block reader needs: the resolved reference frames (Y'CbCr buffers
/// for LAST / GOLDEN / ALTREF), the inter mode read params, and the
/// current frame's aligned luma dimensions for the [`ScaleFactors`]
/// build inside [`decode_and_reconstruct_inter_block`].
#[derive(Clone, Debug)]
pub struct InterFrameDecodeParams<'a> {
    /// Shared base (mi_cols / tx_mode / segmentation / quant).
    pub base: FrameDecodeParams,
    /// Frame's aligned luma width (matches `cur_y_width` in
    /// [`InterBlockInputs`]).
    pub cur_y_width: u32,
    /// Frame's aligned luma height (matches `cur_y_height` in
    /// [`InterBlockInputs`]).
    pub cur_y_height: u32,
    /// Per-logical-ref reference frames (`[LAST, GOLDEN, ALTREF]`).
    /// `None` slots indicate the corresponding pool entry is empty —
    /// libvpx treats those as bitstream errors, but for now we let the
    /// per-block reader either avoid the ref (single-ref blocks pinning
    /// LAST) or panic if a missing ref is selected.
    pub refs: [Option<&'a Yv12Buffer>; 3],
    /// Inter mode read params (sign bias / compound fixed/var refs /
    /// reference_mode / interp_filter / allow_high_precision_mv).
    pub inter_params: InterModeReadParams,
}

/// Run the per-tile decode loop for an inter frame. Mirrors
/// [`decode_tile_intra`] but threads the inter ref pool + per-frame
/// inter params and dispatches per-block to [`decode_inter_block`].
///
/// # Errors
///
/// Returns [`TileDecodeError::InvalidTilePayload`] if the tile bytes
/// don't open with a valid bool-coder marker bit.
pub fn decode_tile_inter(
    tile_data: &[u8],
    bounds: TileBounds,
    params: &InterFrameDecodeParams<'_>,
    fc: &FrameContext,
    dst_frame: &mut Yv12Buffer,
    mut mv_counts: Option<&mut rvp9_core::entropy::mv_probs::NmvCounts>,
    mut coef_counts: Option<&mut rvp9_core::entropy::adapt::CoefCountsBank>,
) -> Result<(), TileDecodeError> {
    let mut reader =
        BoolReader::new(tile_data).map_err(|_| TileDecodeError::InvalidTilePayload)?;
    let tile_mi_cols = bounds.mi_col_end - bounds.mi_col_start;
    let tile_mi_rows = bounds.mi_row_end - bounds.mi_row_start;
    let mut decode_ctx = TileDecodeCtx::new_with_grid(tile_mi_cols, tile_mi_rows);
    let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
    let mut part_ctx = PartitionCtx::new(tile_mi_cols);

    let mut mi_row = bounds.mi_row_start;
    while mi_row < bounds.mi_row_end {
        decode_ctx.reset_left();
        coef_ctx.reset_left_coef();
        part_ctx.reset_left();

        let mut mi_col = bounds.mi_col_start;
        while mi_col < bounds.mi_col_end {
            decode_inter_partition(
                &mut reader,
                fc,
                params,
                &bounds,
                &mut decode_ctx,
                &mut coef_ctx,
                &mut part_ctx,
                dst_frame,
                mi_row,
                mi_col,
                BlockSize::Block64x64,
                4,
                mv_counts.as_deref_mut(),
                coef_counts.as_deref_mut(),
            );
            mi_col += MI_BLOCK_SIZE;
        }
        mi_row += MI_BLOCK_SIZE;
    }
    Ok(())
}

/// Recursively decode one partition starting at `(mi_row, mi_col)`
/// with parent block-size `bsize`. Mirrors libvpx `decode_partition`
/// (`vp9_decodeframe.c:1171-1225`) for inter frames — partition probs
/// come from `fc.partition_prob[ctx]` (3 entries) instead of the
/// keyframe table.
#[allow(clippy::too_many_arguments)]
fn decode_inter_partition(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    params: &InterFrameDecodeParams<'_>,
    bounds: &TileBounds,
    decode_ctx: &mut TileDecodeCtx,
    coef_ctx: &mut TileCoefCtx,
    part_ctx: &mut PartitionCtx,
    dst_frame: &mut Yv12Buffer,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    n4x4_l2: u8,
    mut mv_counts: Option<&mut rvp9_core::entropy::mv_probs::NmvCounts>,
    mut coef_counts: Option<&mut rvp9_core::entropy::adapt::CoefCountsBank>,
) {
    if mi_row >= params.base.mi_rows || mi_col >= params.base.mi_cols {
        return;
    }

    let n8x8_l2 = n4x4_l2.saturating_sub(1);
    let num_8x8_wh: usize = 1 << n8x8_l2;
    let hbs = num_8x8_wh >> 1;
    let has_rows = mi_row + hbs < params.base.mi_rows;
    let has_cols = mi_col + hbs < params.base.mi_cols;

    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let bsl = n8x8_l2 as usize;
    let ctx = part_ctx.context(mi_row_in_tile, mi_col_in_tile, bsl);

    // Inter frames: per-frame `partition_prob[ctx]` (3 entries) drives
    // the partition tree, NOT the static keyframe table.
    let probs = &fc.partition_prob[ctx];
    let partition = if has_rows && has_cols {
        read_partition(reader, probs)
    } else if !has_rows && has_cols {
        if reader.read(probs[1]) != 0 {
            PartitionType::Split
        } else {
            PartitionType::Horz
        }
    } else if has_rows && !has_cols {
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
        decode_inter_block(
            reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
            mi_row, mi_col, sub,
            mv_counts.as_deref_mut(), coef_counts.as_deref_mut(),
        );
    } else {
        match partition {
            PartitionType::None => {
                decode_inter_block(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                    mi_row, mi_col, sub,
                    mv_counts.as_deref_mut(), coef_counts.as_deref_mut(),
                );
            }
            PartitionType::Horz => {
                decode_inter_block(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                    mi_row, mi_col, sub,
                    mv_counts.as_deref_mut(), coef_counts.as_deref_mut(),
                );
                if has_rows {
                    decode_inter_block(
                        reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                        mi_row + hbs, mi_col, sub,
                        mv_counts.as_deref_mut(), coef_counts.as_deref_mut(),
                    );
                }
            }
            PartitionType::Vert => {
                decode_inter_block(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                    mi_row, mi_col, sub,
                    mv_counts.as_deref_mut(), coef_counts.as_deref_mut(),
                );
                if has_cols {
                    decode_inter_block(
                        reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                        mi_row, mi_col + hbs, sub,
                        mv_counts.as_deref_mut(), coef_counts.as_deref_mut(),
                    );
                }
            }
            PartitionType::Split => {
                decode_inter_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row, mi_col, sub, n8x8_l2,
                    mv_counts.as_deref_mut(), coef_counts.as_deref_mut(),
                );
                decode_inter_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row, mi_col + hbs, sub, n8x8_l2,
                    mv_counts.as_deref_mut(), coef_counts.as_deref_mut(),
                );
                decode_inter_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row + hbs, mi_col, sub, n8x8_l2,
                    mv_counts.as_deref_mut(), coef_counts.as_deref_mut(),
                );
                decode_inter_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row + hbs, mi_col + hbs, sub, n8x8_l2,
                    mv_counts, coef_counts,
                );
            }
        }
    }

    if bsize as u8 >= BlockSize::Block8x8 as u8
        && (bsize == BlockSize::Block8x8 || partition != PartitionType::Split)
    {
        part_ctx.update(mi_row_in_tile, mi_col_in_tile, sub, num_8x8_wh);
    }
}

/// Decode one inter-frame leaf coding block at `(mi_row, mi_col)`.
/// Mirrors libvpx `read_inter_frame_mode_info` + the inter arm of
/// `decode_block` (`vp9_decodeframe.c:911-1027`):
///
/// 1. `read_inter_segment_id` (only `update_map == true` path wired
///    today; the temporal-update path is stubbed as zero).
/// 2. `read_skip(ctx)` against `fc.skip_probs`.
/// 3. `read_is_inter_block(ctx)` against `fc.intra_inter_prob` to
///    classify intra-in-inter vs true inter.
/// 4. `read_tx_size(allow_select = !skip || !inter)` against
///    `fc.tx_probs`.
/// 5. If inter: [`decode_and_reconstruct_inter_block`] (D.2).
/// 6. If intra-in-inter: deferred — returns Ok with no pixels written
///    today (D.3.a scaffold). Future work surfaces D-XXX.
#[allow(clippy::too_many_arguments)]
fn decode_inter_block(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    params: &InterFrameDecodeParams<'_>,
    bounds: &TileBounds,
    decode_ctx: &mut TileDecodeCtx,
    coef_ctx: &mut TileCoefCtx,
    dst_frame: &mut Yv12Buffer,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    mv_counts: Option<&mut rvp9_core::entropy::mv_probs::NmvCounts>,
    coef_counts: Option<&mut rvp9_core::entropy::adapt::CoefCountsBank>,
) {
    let mi_row_in_tile = mi_row - bounds.mi_row_start;
    let mi_col_in_tile = mi_col - bounds.mi_col_start;
    let mi_row_in_sb = mi_row_in_tile & MI_MASK;

    // 1. segment_id — same as keyframe path (Stage 3 scope only wires
    //    `update_map = true`; the `temporal_update` / inheritance from
    //    last frame's seg map are out of scope).
    let segment_id = if params.base.segmentation_update_map {
        read_segment_id(reader, &params.base.seg_tree_probs)
    } else {
        0
    };

    // 2. skip flag.
    let above = decode_ctx.above(mi_col_in_tile).copied();
    let left = decode_ctx.left(mi_row_in_sb).copied();
    let skip_ctx = usize::from(above.is_some_and(|m| m.skip))
        + usize::from(left.is_some_and(|m| m.skip));
    let skip = read_skip(reader, &fc.skip_probs, skip_ctx);

    // 3. is_inter classification.
    let intra_inter_ctx = get_intra_inter_context(above.as_ref(), left.as_ref());
    let is_inter = read_is_inter(reader, &fc.intra_inter_prob, intra_inter_ctx);

    // 4. tx_size. allow_select gate matches libvpx
    //    `vp9_decodemv.c:800`: `!mi->skip || !inter_block`.
    let allow_select = (!skip || !is_inter) && !params.base.lossless;
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
        params.base.tx_mode,
        max_tx,
        allow_select,
        tx_ctx,
    );

    // 5. dispatch to inter vs intra mode-info reader.
    let bw_mi = usize::from(bsize.n_8x8_wide());
    let bh_mi = usize::from(bsize.n_8x8_high());
    let x_mis = bw_mi.min(params.base.mi_cols - mi_col);
    let y_mis = bh_mi.min(params.base.mi_rows - mi_row);

    if is_inter {
        // D.3.b: read inter-block mode info, then drive per-plane
        // MC + residual decode via D.2's `reconstruct_inter_block`.
        // Phase D.5 (2026-05-18) replaces the legacy buffer-safe MV
        // clamp with `build_inter_predictor_with_mc_border`, which
        // handles any MV (including ones pointing well past the
        // static border) via a scratch-buffer fallback with libvpx-
        // style edge replication. The on-bitstream MV bounds are
        // already enforced in `read_inter_block_mode_info` via
        // `clamp_mv2` (`vp9_mvref_common.h:20-25`).
        let mut mi = ModeInfo {
            block_size: bsize,
            tx_size,
            skip,
            segment_id,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NearestMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: params.inter_params.interp_filter,
            },
            bmi: [BmiInfo::default(); 4],
        };

        let bounds_block = BlockBounds::new(
            mi_row as i32,
            mi_col as i32,
            params.base.mi_rows as i32,
            params.base.mi_cols as i32,
            bw_mi as i32,
            bh_mi as i32,
        );

        // 1. Read mode info — fills mi.mode (ref_frames / inter_mode /
        //    MVs / interp_filter) and mi.bmi. `mv_counts` (when Some)
        //    accumulates the NEW-MV component reads for backward
        //    adaptation (`vp9_adapt_mv_probs`).
        {
            let lookup = TileNeighbourLookup::new(
                decode_ctx,
                mi_row_in_tile as i32,
                mi_col_in_tile as i32,
            );
            read_inter_block_mode_info(
                reader,
                fc,
                mv_counts,
                &params.inter_params,
                &lookup,
                bsize,
                &bounds_block,
                None,
                &mut mi,
            );
        }

        // 2. Resolve per-block refs against params.refs (LAST=0,
        //    GOLDEN=1, ALTREF=2) and the decoded mi.ref_frames pair.
        let block_refs = match mi.mode {
            BlockMode::Inter { ref_frames, .. } => {
                let resolve = |rf: RefFrame| -> Option<&Yv12Buffer> {
                    match rf {
                        RefFrame::LastFrame => params.refs[0],
                        RefFrame::GoldenFrame => params.refs[1],
                        RefFrame::AltrefFrame => params.refs[2],
                        _ => None,
                    }
                };
                [resolve(ref_frames[0]), resolve(ref_frames[1])]
            }
            _ => unreachable!("read_inter_block_mode_info must populate Inter"),
        };

        // 3. Per-plane recon. Skip if the primary ref is unresolvable
        //    (block leaves zero pixels; MD5 mismatch surfaces).
        if block_refs[0].is_some() {
            let dequant_y = plane_dequant(&params.base, 0, segment_id);
            let dequant_u = plane_dequant(&params.base, 1, segment_id);
            let dequant_v = plane_dequant(&params.base, 2, segment_id);
            let inputs = InterBlockInputs {
                cur_y_width: params.cur_y_width,
                cur_y_height: params.cur_y_height,
                refs: block_refs,
                dequant_per_plane: [dequant_y, dequant_u, dequant_v],
                lossless: params.base.lossless,
            };
            reconstruct_inter_block(
                reader,
                fc,
                coef_counts,
                coef_ctx,
                &mi,
                mi_row,
                mi_col,
                mi_row_in_tile,
                mi_col_in_tile,
                mi_row_in_sb,
                &inputs,
                dst_frame,
            );
        }

        decode_ctx.write_block(mi_row_in_tile, mi_row_in_sb, mi_col_in_tile, x_mis, y_mis, &mi);
    } else {
        // Intra-in-inter-frame path (D.4). libvpx
        // `read_intra_block_mode_info` (`vp9_decodemv.c:354-387`) reads
        // y_mode (per-`bmi` for sub-8) then uv_mode, using
        // `fc.y_mode_prob` (4-row size-group table) NOT the keyframe
        // `vp9_kf_y_mode_prob[above][left]` 2-D conditional. Pixel
        // recon uses the same per-plane TX loop as keyframe intra,
        // factored into `reconstruct_intra_planes`.
        let (y_mode, uv_mode, bmi) = read_intra_in_inter_mode_info(reader, fc, bsize);

        let mi = ModeInfo {
            block_size: bsize,
            tx_size,
            skip,
            segment_id,
            seg_id_predicted: false,
            mode: BlockMode::Intra { y_mode, uv_mode },
            bmi,
        };
        decode_ctx.write_block(mi_row_in_tile, mi_row_in_sb, mi_col_in_tile, x_mis, y_mis, &mi);

        let have_above_block = above.is_some();
        let have_left_block = left.is_some();
        let dequant_y = plane_dequant(&params.base, 0, segment_id);
        let dequant_u = plane_dequant(&params.base, 1, segment_id);
        let dequant_v = plane_dequant(&params.base, 2, segment_id);

        reconstruct_intra_planes(
            reader,
            fc,
            coef_ctx,
            dst_frame,
            &mi,
            y_mode,
            uv_mode,
            &bmi,
            mi_row,
            mi_col,
            mi_row_in_tile,
            mi_col_in_tile,
            mi_row_in_sb,
            have_above_block,
            have_left_block,
            dequant_y,
            dequant_u,
            dequant_v,
            params.base.lossless,
            coef_counts,
        );
    }
}

/// Clamp every active MV in `mi` so the MC kernel's `src_base`
/// arithmetic in [`reconstruct_inter_block`] cannot wrap. Computes
/// the safe Q3 luma MV range from the block's frame-pixel position +
/// the decoder's `DEC_BORDER` (= 32 px), then clamps each component.
///
/// libvpx handles wide-MV cases via `extend_mc_border`
/// (`vp9_reconinter.c:154-205`) which copies reference pixels into a
/// scratch buffer. As of Phase D.5 (2026-05-18) the equivalent
/// scratch fallback lives in
/// `reconstruct_inter::build_inter_predictor_with_mc_border`, so this
/// helper is no longer wired into the recon pipeline. We retain it
/// here as a fallback option (e.g. a `#[cfg(feature = "no-mc-border")]`
/// build flag, or a future correctness-debug switch) — see the Phase
/// D.5 design note in `notes/lld/motion-estimation.md`. The clamp is
/// tighter than libvpx's `clamp_mv_to_umv_border_sb` and produces MD5
/// mismatches when upstream needs the border-extend path; do NOT
/// re-wire it as the primary path.
///
/// We compute separate per-plane bounds (luma + chroma) and take the
/// intersection, since the chroma plane has half-width / half-height
/// and is the tighter constraint for some block positions / frame
/// sizes (notably the small 8×8 conformance vectors).
#[allow(dead_code)] // Retired Phase D.5; see `build_inter_predictor_with_mc_border`.
fn clamp_block_mvs(
    mi: &mut ModeInfo,
    mi_row: usize,
    mi_col: usize,
    bw_mi: usize,
    bh_mi: usize,
    cur_y_width: usize,
    cur_y_height: usize,
) {
    // libvpx `DEC_BORDER_IN_PIXELS` (`yv12config.h:27`).
    const BORDER_PX: i32 = 32;
    // libvpx `VP9_INTERP_EXTEND` (`vp9_reconinter.h:33`).
    const INTERP_EXTEND_PX: i32 = 4;

    // Per-plane safe Q3 MV bounds. `int_x` in plane units is
    // `block_x_plane + (mv >> (3 + ss))`, so the Q3-MV-equivalent of
    // each `int_x` constraint scales by `(1 << (3 + ss))` = 8 for
    // luma, 16 for chroma. We take the intersection of the per-plane
    // bounds across planes 0..=2 (Y, U, V).
    let bw_px_luma = (bw_mi as i32) * 8;
    let bh_px_luma = (bh_mi as i32) * 8;
    let block_x_luma = (mi_col as i32) * 8;
    let block_y_luma = (mi_row as i32) * 8;

    let mut min_col_q3 = i32::MIN / 2;
    let mut max_col_q3 = i32::MAX / 2;
    let mut min_row_q3 = i32::MIN / 2;
    let mut max_row_q3 = i32::MAX / 2;

    // Pre-shift adjustment: reconstruct_inter.rs offsets `src` by
    // `-3 rows, -3 cols` so the convolve8 2-D filter footprint starts
    // at the right place. That moves the effective minimum int_x from
    // `-border` to `-border + 3`, and the effective max upper limit
    // tightens by 3 on the read side: total read extent is
    // `(intermediate_height = bh + 7)` rows starting from `int_y - 3`,
    // so the last row touched is `int_y + bh + 3`. We need
    // `border + int_y + bh + 3 <= plane_h + 2*border`, i.e.
    // `int_y <= plane_h + border - bh - 3`.
    const PRESHIFT: i32 = 3;
    const FILTER_TAIL: i32 = 4; // 8-tap filter trailing taps past int base.

    for ss in 0..=1 {
        // ss=0 → luma; ss=1 → 4:2:0 chroma.
        let shift: i32 = 3 + ss;
        let scale = 1i32 << shift; // 8 for luma, 16 for chroma
        let plane_w = (cur_y_width as i32) >> ss;
        let plane_h = (cur_y_height as i32) >> ss;
        let plane_border = BORDER_PX >> ss;
        let plane_bw = bw_px_luma >> ss;
        let plane_bh = bh_px_luma >> ss;
        let plane_block_x = block_x_luma >> ss;
        let plane_block_y = block_y_luma >> ss;

        // Min: `border + int_x - PRESHIFT >= 0`.
        let lo_int = -plane_border + PRESHIFT - plane_block_x;
        // Max: `border + int_x - PRESHIFT + bw + (FILTER_TAIL + PRESHIFT) <= plane_w + 2*border`,
        // i.e. `int_x <= plane_w + plane_border - bw - FILTER_TAIL - plane_block_x`.
        let _ = INTERP_EXTEND_PX; // kept for documentation parity with libvpx
        let hi_int =
            plane_w + plane_border - plane_bw - FILTER_TAIL - plane_block_x;
        let lo_col = lo_int * scale;
        let hi_col = hi_int * scale + (scale - 1);
        let lo_int_row = -plane_border + PRESHIFT - plane_block_y;
        let hi_int_row =
            plane_h + plane_border - plane_bh - FILTER_TAIL - plane_block_y;
        let lo_row = lo_int_row * scale;
        let hi_row = hi_int_row * scale + (scale - 1);

        min_col_q3 = min_col_q3.max(lo_col);
        max_col_q3 = max_col_q3.min(hi_col);
        min_row_q3 = min_row_q3.max(lo_row);
        max_row_q3 = max_row_q3.min(hi_row);
    }

    let clamp_to_i16 = |v: i32, lo: i32, hi: i32| -> i16 {
        v.clamp(lo, hi)
            .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
    };

    let clamp = |mv: Mv| -> Mv {
        Mv {
            col: clamp_to_i16(i32::from(mv.col), min_col_q3, max_col_q3),
            row: clamp_to_i16(i32::from(mv.row), min_row_q3, max_row_q3),
        }
    };

    match &mut mi.mode {
        BlockMode::Inter { mvs, .. } => {
            for slot in mvs.iter_mut() {
                *slot = clamp(*slot);
            }
        }
        BlockMode::Intra { .. } => {}
    }
    for bmi in mi.bmi.iter_mut() {
        for slot in bmi.as_mv.iter_mut() {
            *slot = clamp(*slot);
        }
    }
}

/// Read the per-block intra-frame mode info inside an inter frame.
/// Mirrors libvpx `read_intra_block_mode_info` (`vp9_decodemv.c:354-387`).
///
/// Probabilities: `fc.y_mode_prob[size_group]` (different table from
/// the keyframe `VP9_KF_Y_MODE_PROB`), and `fc.uv_mode_prob[y_mode]`.
/// Sub-8 reads pass `size_group = 0` per libvpx — the BLOCK_4X4 /
/// BLOCK_4X8 / BLOCK_8X4 arms read 4 / 2 / 2 per-`bmi` modes using
/// the `fc.y_mode_prob[0]` row.
///
/// Returns `(y_mode, uv_mode, bmi)`. For sub-8 sizes the `bmi[0..4]`
/// array is populated with the per-4×4 mode duplicates exactly as
/// libvpx does (`mi->bmi[i].as_mode`); for >= 8×8 only the parent
/// `y_mode` is meaningful and `bmi` is zero-initialised.
fn read_intra_in_inter_mode_info(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    bsize: BlockSize,
) -> (IntraMode, IntraMode, [BmiInfo; 4]) {
    // libvpx `size_group_lookup` (`vp9_common_data.c:14-19`).
    let size_group = match bsize {
        BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4
        | BlockSize::Block8x8 => 0,
        BlockSize::Block8x16 | BlockSize::Block16x8 | BlockSize::Block16x16 => 1,
        BlockSize::Block16x32 | BlockSize::Block32x16 | BlockSize::Block32x32 => 2,
        BlockSize::Block32x64 | BlockSize::Block64x32 | BlockSize::Block64x64 => 3,
    };

    let mut bmi = [BmiInfo::default(); 4];

    // libvpx `read_intra_block_mode_info` arms — for sub-8 sizes the
    // size_group passed to `read_intra_mode_y` is hard-coded to 0,
    // NOT `size_group_lookup[bsize]`. See `vp9_decodemv.c:363-374`.
    let y_mode = match bsize {
        BlockSize::Block4x4 => {
            for slot in bmi.iter_mut() {
                let m = read_intra_mode(reader, &fc.y_mode_prob[0]);
                slot.as_mode = m as u8;
            }
            // libvpx: `mi->mode = mi->bmi[3].as_mode`.
            IntraMode::from_u8(bmi[3].as_mode).expect("legal mode")
        }
        BlockSize::Block4x8 => {
            let m0 = read_intra_mode(reader, &fc.y_mode_prob[0]);
            bmi[0].as_mode = m0 as u8;
            bmi[2].as_mode = m0 as u8;
            let m1 = read_intra_mode(reader, &fc.y_mode_prob[0]);
            bmi[1].as_mode = m1 as u8;
            bmi[3].as_mode = m1 as u8;
            m1
        }
        BlockSize::Block8x4 => {
            let m0 = read_intra_mode(reader, &fc.y_mode_prob[0]);
            bmi[0].as_mode = m0 as u8;
            bmi[1].as_mode = m0 as u8;
            let m2 = read_intra_mode(reader, &fc.y_mode_prob[0]);
            bmi[2].as_mode = m2 as u8;
            bmi[3].as_mode = m2 as u8;
            m2
        }
        _ => read_intra_mode(reader, &fc.y_mode_prob[size_group]),
    };

    let uv_mode = read_intra_mode(reader, &fc.uv_mode_prob[y_mode as usize]);
    (y_mode, uv_mode, bmi)
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

    /// D.4 — `read_intra_in_inter_mode_info` for a ≥ 8×8 block uses
    /// `fc.y_mode_prob[size_group_lookup[bsize]]` (NOT the keyframe
    /// `VP9_KF_Y_MODE_PROB[above][left]` 2-D conditional). Encode a
    /// bitstream that drives the inter-intra Y tree to `IntraMode::H`
    /// against the `default_if_y_probs` size-group-3 row, then verify
    /// the reader picks the same mode (and that the keyframe-table
    /// decode would NOT match — the prob tables differ).
    #[test]
    fn intra_in_inter_uses_default_if_y_probs_for_64x64() {
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::mode_info::INTRA_MODE_TREE;

        fn subtree_contains(node: i8, target: IntraMode) -> bool {
            if node <= 0 {
                (-node) as u8 == target as u8
            } else {
                let i = node as usize;
                subtree_contains(INTRA_MODE_TREE[i], target)
                    || subtree_contains(INTRA_MODE_TREE[i + 1], target)
            }
        }
        fn encode_intra_mode(target: IntraMode, probs: &[u8; 9], w: &mut BoolWriter) {
            let mut i: usize = 0;
            loop {
                let prob = probs[i >> 1];
                let left = INTRA_MODE_TREE[i];
                let right = INTRA_MODE_TREE[i + 1];
                let bit = if subtree_contains(left, target) { 0 } else { 1 };
                w.write(bit, prob);
                let next = if bit == 0 { left } else { right };
                if next <= 0 {
                    break;
                }
                i = next as usize;
            }
        }

        let fc = FrameContext::new_default();
        // BLOCK_64X64 → size_group_lookup = 3.
        let y_probs = fc.y_mode_prob[3];

        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        encode_intra_mode(IntraMode::H, &y_probs, &mut w);
        // uv_mode uses fc.uv_mode_prob[y_mode]; encode H here too for
        // determinism.
        encode_intra_mode(IntraMode::H, &fc.uv_mode_prob[IntraMode::H as usize], &mut w);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let n = w.stop_encode().expect("encode");
        buf.truncate(n);

        let mut r = BoolReader::new(&buf).expect("init");
        let (y_mode, uv_mode, bmi) =
            read_intra_in_inter_mode_info(&mut r, &fc, BlockSize::Block64x64);
        assert_eq!(y_mode, IntraMode::H, "y_mode round-trip via default_if_y_probs");
        assert_eq!(uv_mode, IntraMode::H);
        // ≥ 8×8: bmi is zero-init (libvpx never writes per-bmi modes).
        for slot in &bmi {
            assert_eq!(slot.as_mode, 0);
        }

        // Negative: the same bitstream decoded via VP9_KF_Y_MODE_PROB
        // (e.g. [above=Dc, left=Dc] row used at frame top-left for
        // keyframes) yields a DIFFERENT mode — confirming this code
        // path picked the inter-intra table, not the kf table.
        let kf_probs = VP9_KF_Y_MODE_PROB[IntraMode::Dc as usize][IntraMode::Dc as usize];
        if kf_probs != y_probs {
            let mut r2 = BoolReader::new(&buf).expect("init");
            let alt = read_intra_mode(&mut r2, &kf_probs);
            assert_ne!(
                alt,
                IntraMode::H,
                "kf_probs decode should diverge from inter-intra path \
                 (else the test does not distinguish the two tables)",
            );
        }
    }

    /// D.4 — sub-8 BLOCK_4X4 reads four per-`bmi` y_mode entries via
    /// `fc.y_mode_prob[0]` (libvpx hard-codes `size_group = 0` for the
    /// per-bmi loop, regardless of bsize). Verify all four 4×4
    /// sub-modes round-trip and `mi->mode` = `bmi[3].as_mode`.
    #[test]
    fn intra_in_inter_block4x4_reads_four_bmi_via_size_group_0() {
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::mode_info::INTRA_MODE_TREE;

        fn subtree_contains(node: i8, target: IntraMode) -> bool {
            if node <= 0 {
                (-node) as u8 == target as u8
            } else {
                let i = node as usize;
                subtree_contains(INTRA_MODE_TREE[i], target)
                    || subtree_contains(INTRA_MODE_TREE[i + 1], target)
            }
        }
        fn encode_intra_mode(target: IntraMode, probs: &[u8; 9], w: &mut BoolWriter) {
            let mut i: usize = 0;
            loop {
                let prob = probs[i >> 1];
                let left = INTRA_MODE_TREE[i];
                let right = INTRA_MODE_TREE[i + 1];
                let bit = if subtree_contains(left, target) { 0 } else { 1 };
                w.write(bit, prob);
                let next = if bit == 0 { left } else { right };
                if next <= 0 {
                    break;
                }
                i = next as usize;
            }
        }

        let fc = FrameContext::new_default();
        let y_probs = fc.y_mode_prob[0];
        let modes = [IntraMode::Dc, IntraMode::V, IntraMode::H, IntraMode::Tm];

        let mut buf = vec![0u8; 128];
        let mut w = BoolWriter::start_encode(&mut buf);
        for m in &modes {
            encode_intra_mode(*m, &y_probs, &mut w);
        }
        // uv_mode follows mi->mode = bmi[3] = Tm.
        encode_intra_mode(IntraMode::Dc, &fc.uv_mode_prob[IntraMode::Tm as usize], &mut w);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let n = w.stop_encode().expect("encode");
        buf.truncate(n);

        let mut r = BoolReader::new(&buf).expect("init");
        let (y_mode, uv_mode, bmi) =
            read_intra_in_inter_mode_info(&mut r, &fc, BlockSize::Block4x4);
        // libvpx: mi->mode = mi->bmi[3].as_mode.
        assert_eq!(y_mode, IntraMode::Tm);
        assert_eq!(uv_mode, IntraMode::Dc);
        assert_eq!(bmi[0].as_mode, IntraMode::Dc as u8);
        assert_eq!(bmi[1].as_mode, IntraMode::V as u8);
        assert_eq!(bmi[2].as_mode, IntraMode::H as u8);
        assert_eq!(bmi[3].as_mode, IntraMode::Tm as u8);
    }

    /// D.4 — BLOCK_8X4 reads two per-`bmi` modes (rows 0+1 share
    /// `bmi[0]`, rows 2+3 share `bmi[2]`), both via
    /// `fc.y_mode_prob[0]`. libvpx `vp9_decodemv.c:371-375`.
    #[test]
    fn intra_in_inter_block8x4_duplicates_bmi_rows() {
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::mode_info::INTRA_MODE_TREE;

        fn subtree_contains(node: i8, target: IntraMode) -> bool {
            if node <= 0 {
                (-node) as u8 == target as u8
            } else {
                let i = node as usize;
                subtree_contains(INTRA_MODE_TREE[i], target)
                    || subtree_contains(INTRA_MODE_TREE[i + 1], target)
            }
        }
        fn encode_intra_mode(target: IntraMode, probs: &[u8; 9], w: &mut BoolWriter) {
            let mut i: usize = 0;
            loop {
                let prob = probs[i >> 1];
                let left = INTRA_MODE_TREE[i];
                let right = INTRA_MODE_TREE[i + 1];
                let bit = if subtree_contains(left, target) { 0 } else { 1 };
                w.write(bit, prob);
                let next = if bit == 0 { left } else { right };
                if next <= 0 {
                    break;
                }
                i = next as usize;
            }
        }

        let fc = FrameContext::new_default();
        let y_probs = fc.y_mode_prob[0];

        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        encode_intra_mode(IntraMode::V, &y_probs, &mut w);
        encode_intra_mode(IntraMode::H, &y_probs, &mut w);
        encode_intra_mode(IntraMode::Dc, &fc.uv_mode_prob[IntraMode::H as usize], &mut w);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let n = w.stop_encode().expect("encode");
        buf.truncate(n);

        let mut r = BoolReader::new(&buf).expect("init");
        let (y_mode, _uv, bmi) =
            read_intra_in_inter_mode_info(&mut r, &fc, BlockSize::Block8x4);
        // libvpx: mi->bmi[0] = bmi[1] = first read; bmi[2] = bmi[3] =
        // second read; mi->mode = bmi[2] (= second read).
        assert_eq!(y_mode, IntraMode::H);
        assert_eq!(bmi[0].as_mode, IntraMode::V as u8);
        assert_eq!(bmi[1].as_mode, IntraMode::V as u8);
        assert_eq!(bmi[2].as_mode, IntraMode::H as u8);
        assert_eq!(bmi[3].as_mode, IntraMode::H as u8);
    }

    /// Stage 3 stretch M.11 — regression: a sub-8 intra left neighbour
    /// whose right-column bmi entry is `DC_PRED` (numeric value `0`)
    /// must be reported as `Dc` by `left_block_mode`, NOT collapsed to
    /// the parent's `y_mode` via an `as_mode != 0` "is-populated"
    /// sentinel. libvpx `get_y_mode` (`vp9_blockd.h:99-101`) branches on
    /// the neighbour's `sb_type`, not on the bmi byte value.
    #[test]
    fn left_block_mode_sub8_neighbour_returns_dc_when_bmi_is_dc() {
        // Left neighbour is sub-8 (Block8x4) with parent y_mode V and
        // bmi[1].as_mode = DC_PRED (= 0). For block=0 the left lookup
        // reads neighbour bmi[block + 1 = 1].
        let mut n_bmi = [BmiInfo::default(); 4];
        n_bmi[0].as_mode = IntraMode::V as u8;
        n_bmi[1].as_mode = IntraMode::Dc as u8; // legit Dc, value 0.
        n_bmi[2].as_mode = IntraMode::V as u8;
        n_bmi[3].as_mode = IntraMode::Dc as u8;
        let left = Some((
            BlockSize::Block8x4,
            BlockMode::Intra {
                y_mode: IntraMode::V,
                uv_mode: IntraMode::Dc,
            },
            n_bmi,
        ));
        let cur_bmi = [BmiInfo::default(); 4];
        assert_eq!(left_block_mode(0, &cur_bmi, left), IntraMode::Dc);
        // For block=2 the left lookup reads neighbour bmi[3] which is
        // also DC_PRED in this fixture.
        assert_eq!(left_block_mode(2, &cur_bmi, left), IntraMode::Dc);
    }

    /// Mirror of the previous test for `above_block_mode` — sub-8
    /// above neighbour whose bottom-row bmi entry is `DC_PRED` must be
    /// reported as `Dc`, not as the parent's `y_mode`.
    #[test]
    fn above_block_mode_sub8_neighbour_returns_dc_when_bmi_is_dc() {
        let mut n_bmi = [BmiInfo::default(); 4];
        n_bmi[0].as_mode = IntraMode::H as u8;
        n_bmi[1].as_mode = IntraMode::H as u8;
        n_bmi[2].as_mode = IntraMode::Dc as u8; // block=0 reads bmi[2].
        n_bmi[3].as_mode = IntraMode::Dc as u8; // block=1 reads bmi[3].
        let above = Some((
            BlockSize::Block4x8,
            BlockMode::Intra {
                y_mode: IntraMode::H,
                uv_mode: IntraMode::Dc,
            },
            n_bmi,
        ));
        let cur_bmi = [BmiInfo::default(); 4];
        assert_eq!(above_block_mode(0, &cur_bmi, above), IntraMode::Dc);
        assert_eq!(above_block_mode(1, &cur_bmi, above), IntraMode::Dc);
    }

    /// A `>= 8×8` neighbour returns the parent `y_mode` regardless of
    /// the bmi contents — libvpx's `get_y_mode` branches on
    /// `sb_type < BLOCK_8X8`. This guards against accidentally reading
    /// the (zero-init) bmi for >=8×8 parents.
    #[test]
    fn left_block_mode_ge_8x8_neighbour_returns_parent_y_mode() {
        let n_bmi = [BmiInfo::default(); 4]; // all zero (would look like Dc).
        let left = Some((
            BlockSize::Block16x16,
            BlockMode::Intra {
                y_mode: IntraMode::V,
                uv_mode: IntraMode::Dc,
            },
            n_bmi,
        ));
        let cur_bmi = [BmiInfo::default(); 4];
        assert_eq!(left_block_mode(0, &cur_bmi, left), IntraMode::V);
        assert_eq!(left_block_mode(2, &cur_bmi, left), IntraMode::V);
    }

    /// An inter neighbour collapses to `Dc` regardless of bmi /
    /// block_size — libvpx's
    /// `if (!left_mi || is_inter_block(left_mi)) return DC_PRED`.
    #[test]
    fn left_block_mode_inter_neighbour_collapses_to_dc() {
        use rvp9_core::bitstream::frame::InterpFilter;
        use rvp9_core::mode_info::{InterMode, Mv, RefFrame};
        let n_bmi = [BmiInfo::default(); 4];
        let left = Some((
            BlockSize::Block8x4,
            BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            n_bmi,
        ));
        let cur_bmi = [BmiInfo::default(); 4];
        assert_eq!(left_block_mode(0, &cur_bmi, left), IntraMode::Dc);
    }
}
