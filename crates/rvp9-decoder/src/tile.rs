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
use rvp9_core::intra_pred::{EdgeBuffers, IntraMode, build_intra_predictors};
use rvp9_core::mode_info::{BlockMode, BmiInfo, ModeInfo, read_intra_mode, read_partition,
    read_segment_id, read_skip, read_tx_size, get_tx_size_context};
use rvp9_core::quant::{ac_quant, dc_quant, get_qindex};
use rvp9_core::transforms::{TxMode, TxSize};

use crate::reconstruct::reconstruct_intra_block;
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
    dst_frame: &mut Yv12Buffer,
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
            reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
            mi_row, mi_col, sub,
        );
    } else {
        match partition {
            PartitionType::None => {
                decode_block(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                    mi_row, mi_col, sub,
                );
            }
            PartitionType::Horz => {
                decode_block(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                    mi_row, mi_col, sub,
                );
                if has_rows {
                    decode_block(
                        reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                        mi_row + hbs, mi_col, sub,
                    );
                }
            }
            PartitionType::Vert => {
                decode_block(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                    mi_row, mi_col, sub,
                );
                if has_cols {
                    decode_block(
                        reader, fc, params, bounds, decode_ctx, coef_ctx, dst_frame,
                        mi_row, mi_col + hbs, sub,
                    );
                }
            }
            PartitionType::Split => {
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row, mi_col, sub, n8x8_l2,
                );
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row, mi_col + hbs, sub, n8x8_l2,
                );
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row + hbs, mi_col, sub, n8x8_l2,
                );
                decode_partition(
                    reader, fc, params, bounds, decode_ctx, coef_ctx, part_ctx,
                    dst_frame, mi_row + hbs, mi_col + hbs, sub, n8x8_l2,
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
        above.map(|m| (m.mode, m.bmi)),
        left.map(|m| (m.mode, m.bmi)),
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

    if mi.skip {
        // libvpx `dec_reset_skip_context` — zero the per-plane above/
        // left coef-context cells covering the block. Equivalent to
        // splatting `(eob > 0) == 0` across the coding-block region.
        reset_skip_context(coef_ctx, &mi, mi_row_in_tile, mi_col_in_tile);
    }

    // Whether per-block coding-block-level neighbours exist. Mirrors
    // libvpx `xd->above_mi != NULL` / `xd->left_mi != NULL` — for
    // keyframes these come from the parent coding block's neighbour
    // grid via the row+col reduction we keep in TileDecodeCtx.
    //
    // NB: read AFTER `write_block` only for tile-edge / above-edge
    // semantics — the values stored above by `write_block` cover the
    // current block's outgoing slot, not its incoming neighbour. So
    // for the incoming neighbour we re-read from `decode_ctx` BUT
    // remember `decode_ctx.above` is the pre-write state for cells
    // outside the just-written span. For a block at `mi_col_in_tile`
    // the row above us comes from `mi_col_in_tile - 1` etc; libvpx
    // checks `xd->above_mi == NULL` based on frame-top semantics, not
    // current-block self-reference.
    //
    // We snapshot the inbound flags here, before the per-plane loop.
    let have_above_block = above.is_some();
    let have_left_block = left.is_some();

    let dequant_y = plane_dequant(params, 0, segment_id);
    let dequant_u = plane_dequant(params, 1, segment_id);
    let dequant_v = plane_dequant(params, 2, segment_id);

    let mut edges = EdgeBuffers::new();
    for plane in 0..3 {
        let ss_x = SUBSAMPLING_X[plane];
        let ss_y = SUBSAMPLING_Y[plane];

        // libvpx `get_uv_tx_size` (`vp9_blockd.h:265-267`):
        // chroma tx_size = min(mi->tx_size, max_txsize_lookup[plane_bsize]).
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
        // libvpx pd->n4_w / n4_h (vp9_decodeframe.c:796-797): for the
        // sub-8 coding-block path, `decode_partition` calls
        // `decode_block(..., subsize, 1, 1)` (vp9_decodeframe.c:1192),
        // so `bw = bh = 1` mi-unit and `n4_w = n4_h = (1 << 1) >> ss = 2 >> ss`
        // regardless of the actual sub-8 subsize (Block4x4 / Block4x8 /
        // Block8x4). The four 4×4 sub-positions are all iterated, with
        // per-bmi modes selected inside the loop. For block_size ≥ 8×8
        // the iteration extent follows `plane_bsize.n_4x4_*()` as usual.
        let (plane_bw_4x4, plane_bh_4x4) = if bsize < BlockSize::Block8x8 {
            (2 >> ss_x, 2 >> ss_y)
        } else {
            (
                plane_bsize.n_4x4_wide() as usize,
                plane_bsize.n_4x4_high() as usize,
            )
        };
        let tx_step_4x4 = 1usize << (plane_tx as usize);

        // Pixel coords of the block's top-left in plane-sample space.
        let plane_px_x = (mi_col * 8) >> ss_x;
        let plane_px_y = (mi_row * 8) >> ss_y;

        // Plane-local block coordinates (4×4-cell base for the
        // coef-context arrays — which are indexed in 4×4 sub-block
        // units, plane-relative).
        let plane_x4_base = (mi_col_in_tile * 2) >> ss_x;
        let plane_y4_in_sb_base = ((mi_row_in_sb) * 2) >> ss_y;

        // Extract plane-mutable view + dimensions for this plane.
        // The temporary `PlaneMut` is dropped before we exit the
        // block, but `data_mut()` re-borrows into a slice that lives
        // for the rest of the per-plane loop. We bind the
        // intermediate to keep its lifetime explicit.
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

        // Iterate TX positions inside the plane's coding block.
        let mut loff = 0usize;
        while loff < plane_bh_4x4 {
            let mut aoff = 0usize;
            while aoff < plane_bw_4x4 {
                // Pixel coordinates of this TX block.
                let tx_px_x = plane_px_x + aoff * 4;
                let tx_px_y = plane_px_y + loff * 4;

                // Skip TX blocks whose top-left is outside the plane
                // (block straddles the right/bottom frame edge).
                if tx_px_x >= plane_width_px || tx_px_y >= plane_height_px {
                    aoff += tx_step_4x4;
                    continue;
                }

                let bw_in_4x4 = plane_bw_4x4; // parent block width in 4×4
                let have_top = loff > 0 || have_above_block;
                let have_left = aoff > 0 || have_left_block;
                let have_right = (aoff + tx_step_4x4) < bw_in_4x4;

                // Resolve per-sub-block Y intra mode for sub-8 (bmi).
                let mode_for_this_tx = if plane == 0 && bsize < BlockSize::Block8x8 {
                    let bi = (loff << 1) | aoff; // libvpx (row << 1) + col
                    IntraMode::from_u8(bmi[bi].as_mode).unwrap_or(plane_intra_mode)
                } else {
                    plane_intra_mode
                };

                // Build edges from already-reconstructed pixels.
                // `plane_data` is borrowed immutably here.
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

                // Reconstruct (predict + decode tokens + IT-add).
                let pred_ctx = edges.pred_ctx(plane_tx.side());
                let dst_offset = tx_px_y * plane_stride + tx_px_x;
                let dst_slice = &mut plane_data[dst_offset..];

                if mi.skip {
                    // libvpx: no token decode; predicted samples are
                    // the final reconstruction. We still need to
                    // write the prediction into `dst`.
                    rvp9_core::intra_pred::intra_predict(
                        mode_for_this_tx,
                        dst_slice,
                        plane_stride,
                        plane_tx.side(),
                        pred_ctx,
                    );
                } else {
                    let seg_eob = plane_tx.n_coeffs();
                    let _eob = reconstruct_intra_block(
                        reader,
                        fc,
                        /* counts = */ None,
                        coef_ctx,
                        plane,
                        plane_x4_base + aoff,
                        plane_y4_in_sb_base + loff,
                        plane_tx,
                        mode_for_this_tx,
                        pred_ctx,
                        dst_slice,
                        plane_stride,
                        seg_eob,
                        plane_dequant_pair,
                        params.lossless && plane_tx == TxSize::Tx4x4,
                    );
                }

                aoff += tx_step_4x4;
            }
            loff += tx_step_4x4;
        }
    }
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
    above: Option<(BlockMode, [BmiInfo; 4])>,
    left: Option<(BlockMode, [BmiInfo; 4])>,
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
    above: Option<(BlockMode, [BmiInfo; 4])>,
    left: Option<(BlockMode, [BmiInfo; 4])>,
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
