//! Per-tile neighbour-context plumbing: the `above` row and `left`
//! column of [`ModeInfo`] descriptors that every per-block decoder
//! reads to compute its prediction contexts.
//!
//! ## Scope (Stage 3 Task 8.2.6, "Wave 2 D")
//!
//! Token-coefficient decode (Wave 3 F), per-block intra-mode read,
//! `read_tx_size`, `read_skip`, and the segmentation/skip
//! prediction paths all consume the above + left neighbour
//! [`ModeInfo`] to compute their boolean-context indices. Before this
//! module landed, [`rvp9_core::mode_info::get_tx_size_context`] was
//! receiving `(None, None)` from every caller because there was no
//! place to read real neighbours from. This module stands up that
//! plumbing.
//!
//! ## Storage model
//!
//! libvpx's `VP9_COMMON` maintains a full 2-D `mi_grid_visible[]`
//! of pointers — one per 8×8 mode-info slot in the entire frame
//! (`vp9_alloccommon.c:vp9_alloc_context_buffers`). Each per-block
//! decoder reads `above_mi = xd->mi[-xd->mi_stride]` (row above) and
//! `left_mi = xd->mi[-1]` (column to the left), where the pointers
//! return `NULL` at frame top / tile left edges
//! (`vp9_onyxc_int.h:set_mi_row_col`).
//!
//! For Task 8.2.6 we use the simpler "above-row + left-column"
//! reduction:
//!
//! - **`above`** — one [`ModeInfo`] slot per 8×8 column across the
//!   tile width. Survives between SB rows; the bottom edge of each
//!   block written into it becomes the "above" neighbour of the
//!   block directly below in the next SB row.
//! - **`left`** — one [`ModeInfo`] slot per 8×8 row in the current SB
//!   row. Reset to `None` at the start of every new SB row (libvpx
//!   `vp9_decodeframe.c:decode_tiles` resets `left_context` between
//!   SB rows; see also `init_macroblockd`).
//!
//! This works for every neighbour-context predicate VP9 currently
//! needs (`tx_size`, `skip`, `is_inter`, `intra_mode_uv`, segment-ID
//! prediction, partition context). The NW (above-left) neighbour
//! used by some libvpx context computations (e.g. `get_pred_context_
//! single_ref_p1`) is NOT preserved by this reduction; if a later
//! Stage-3 sub-task needs NW we extend this struct to a full 2-D
//! grid then.
//!
//! ## Frame-edge / tile-edge semantics
//!
//! - The accessors return `None` at frame top (no above-row entry
//!   has been written for that column) and at tile left edge (no
//!   block at `mi_col == 0` in the tile). This matches libvpx
//!   `up_available = 0` / `left_available = 0` exactly (see
//!   `vp9_pred_common.h:156-171:get_tx_size_context`). The
//!   `(None, None)` arm in [`rvp9_core::mode_info::get_tx_size_context`]
//!   collapses both sides to `max_tx_size`, equivalent to libvpx's
//!   `xd->above_mi == NULL && xd->left_mi == NULL` path.
//!
//! ## Where this gets called from
//!
//! Today: nowhere — the per-block decode loop (Wave 3 K) is the
//! first real caller. The unit tests below exercise the data
//! structure directly. Wave 3 K will instantiate one
//! [`TileDecodeCtx`] per tile, walk SBs row-by-row, and at each
//! 8×8 block call:
//!
//! ```ignore
//! let above = ctx.above(mi_col);
//! let left  = ctx.left(mi_row_in_sb);
//! let tx_ctx = get_tx_size_context(
//!     above.map(|mi| (mi.tx_size, mi.skip)),
//!     left .map(|mi| (mi.tx_size, mi.skip)),
//!     max_tx_size,
//! );
//! // ... decode block ...
//! ctx.write_block(mi_row_in_sb, mi_col, n_8x8_wide, n_8x8_high, &decoded_mi);
//! ```

use rvp9_core::mode_info::ModeInfo;

/// libvpx `MI_BLOCK_SIZE` (`vp9_enums.h:25`). Each 64×64 superblock
/// covers an 8×8 mode-info grid.
const MI_BLOCK_SIZE: usize = 8;

/// Per-tile neighbour-context state.
///
/// Holds the "above row" + "left column" of [`ModeInfo`] descriptors
/// the per-block decoder consults to compute its prediction
/// contexts. See the module docs for the design rationale.
///
/// One instance per tile. The `above` row persists across SB rows
/// within the tile; the `left` column resets at the start of each
/// new SB row via [`TileDecodeCtx::reset_left`].
#[derive(Clone, Debug)]
pub struct TileDecodeCtx {
    /// Above-row neighbour storage — one slot per 8×8 column in the
    /// tile width. `None` means "no block decoded yet in that
    /// column" (i.e. we're still on the topmost SB row of the tile)
    /// — which is also the libvpx `above_mi == NULL` case at frame
    /// top.
    above: Vec<Option<ModeInfo>>,
    /// Left-column neighbour storage — one slot per 8×8 row within
    /// the currently-active SB row. `None` means "no block to the
    /// left in this row yet" (libvpx's `left_mi == NULL` at tile
    /// left edge).
    left: Vec<Option<ModeInfo>>,
}

impl TileDecodeCtx {
    /// Create a fresh per-tile context covering `tile_mi_cols`
    /// columns of 8×8 blocks. All slots start `None`.
    ///
    /// `tile_mi_cols` is the tile's width measured in mi-units (8-px
    /// units), i.e. `tile.mi_col_end - tile.mi_col_start`. The
    /// `left` column always has length `MI_BLOCK_SIZE = 8` (one per
    /// 8×8 row inside the current SB row).
    ///
    /// # Panics
    ///
    /// If `tile_mi_cols == 0` — every legal VP9 tile spans at
    /// least one SB column.
    #[must_use]
    pub fn new(tile_mi_cols: usize) -> Self {
        assert!(
            tile_mi_cols > 0,
            "tile must span at least one 8×8 mode-info column"
        );
        Self {
            above: vec![None; tile_mi_cols],
            left: vec![None; MI_BLOCK_SIZE],
        }
    }

    /// Number of 8×8 columns covered by this tile.
    #[must_use]
    pub fn tile_mi_cols(&self) -> usize {
        self.above.len()
    }

    /// Read the above-row neighbour at tile-relative column
    /// `mi_col`. Returns `None` at frame top (no block has been
    /// written into that column yet — libvpx `up_available = 0`).
    ///
    /// `mi_col` is the column index within the tile (0-based, in
    /// mi-units). Values `>= tile_mi_cols()` are out of range and
    /// return `None` (defensive; callers shouldn't request these).
    #[must_use]
    pub fn above(&self, mi_col: usize) -> Option<&ModeInfo> {
        self.above.get(mi_col)?.as_ref()
    }

    /// Read the left-column neighbour at SB-row-relative row
    /// `mi_row_in_sb` (0..8). Returns `None` at the tile's left
    /// edge (no block has been written into the left column yet —
    /// libvpx `left_available = 0`).
    ///
    /// `mi_row_in_sb` is the row within the current SB (0-based,
    /// in mi-units). Values `>= MI_BLOCK_SIZE` are out of range and
    /// return `None`.
    #[must_use]
    pub fn left(&self, mi_row_in_sb: usize) -> Option<&ModeInfo> {
        self.left.get(mi_row_in_sb)?.as_ref()
    }

    /// Reset the left-column slots at the start of a new SB row.
    /// Mirrors libvpx `vp9_decodeframe.c:decode_tiles` clearing
    /// `left_context` between SB rows.
    pub fn reset_left(&mut self) {
        for slot in &mut self.left {
            *slot = None;
        }
    }

    /// Write a decoded block's [`ModeInfo`] into the above-row +
    /// left-column slots it covers, so the next block can read it
    /// as its above / left neighbour.
    ///
    /// Parameters:
    /// - `mi_row_in_sb`: top-left row of the block, in mi-units,
    ///   relative to the current SB row (0..8).
    /// - `mi_col`: top-left column of the block, in mi-units,
    ///   relative to the tile (0..`tile_mi_cols()`).
    /// - `n_8x8_wide`, `n_8x8_high`: block size in mi-units (i.e.
    ///   `bsize.n_8x8_wide()` / `n_8x8_high()`). For sub-8 partitions
    ///   libvpx clamps both to 1 (see [`rvp9_core::block::BlockSize::
    ///   n_8x8_wide`]) — we follow suit, so a 4×4 block still writes
    ///   one (above, left) slot.
    /// - `mi`: the decoded [`ModeInfo`] to deposit into every (row,
    ///   col) slot the block covers.
    ///
    /// Libvpx writes the same `MODE_INFO*` to every covered cell in
    /// the 2-D `mi_grid_visible`. For the row+column reduction we
    /// only need to write:
    /// - The **bottom-row** projection into `above[mi_col..mi_col+w]`
    ///   (this is what the block directly below in the NEXT SB row
    ///   will read as its `above_mi`).
    /// - The **right-column** projection into `left[mi_row..
    ///   mi_row+h]` (this is what the block directly to the right
    ///   in this SB row will read as its `left_mi`).
    ///
    /// We clamp writes to the tile bounds; out-of-bounds slots are
    /// silently dropped. (libvpx's `x_mis`/`y_mis` clamp is the
    /// equivalent guard at `vp9_decodeframe.c:920`.)
    pub fn write_block(
        &mut self,
        mi_row_in_sb: usize,
        mi_col: usize,
        n_8x8_wide: usize,
        n_8x8_high: usize,
        mi: &ModeInfo,
    ) {
        // Above-row write: the block's bottom row is the row that
        // the block directly below in the NEXT SB row will see as
        // its above neighbour. For row/col reduction we collapse
        // this to the same `mi` value across the block's width.
        let col_end = (mi_col + n_8x8_wide).min(self.above.len());
        for slot in &mut self.above[mi_col..col_end] {
            *slot = Some(*mi);
        }

        // Left-column write: the block's right edge is what the
        // block directly to the right in this SB row will see as
        // its left neighbour.
        let row_end = (mi_row_in_sb + n_8x8_high).min(self.left.len());
        for slot in &mut self.left[mi_row_in_sb..row_end] {
            *slot = Some(*mi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::bitstream::InterpFilter;
    use rvp9_core::block::BlockSize;
    use rvp9_core::intra_pred::IntraMode;
    use rvp9_core::mode_info::{BlockMode, InterMode, Mv, RefFrame};
    use rvp9_core::transforms::TxSize;

    /// Build a representative intra [`ModeInfo`] for testing. The
    /// fields are arbitrary but distinct so equality checks can
    /// detect cross-contamination between slots.
    fn mk_intra(tx_size: TxSize, skip: bool, segment_id: u8) -> ModeInfo {
        ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size,
            skip,
            segment_id,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    /// A 16×16-shaped intra [`ModeInfo`] (n_8x8_wide=2, n_8x8_high=2).
    fn mk_intra_16x16(tx_size: TxSize, skip: bool) -> ModeInfo {
        ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size,
            skip,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::V,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    /// A 16×8 inter block (n_8x8_wide=2, n_8x8_high=1).
    fn mk_inter_16x8() -> ModeInfo {
        ModeInfo {
            block_size: BlockSize::Block16x8,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NearestMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        }
    }

    /// Fresh context: every slot is `None`. This is the libvpx
    /// `(above_mi == NULL, left_mi == NULL)` case at the
    /// frame top-left corner.
    #[test]
    fn fresh_ctx_returns_none_for_corner() {
        let ctx = TileDecodeCtx::new(8);
        assert_eq!(ctx.tile_mi_cols(), 8);
        assert!(ctx.above(0).is_none(), "top-left above");
        assert!(ctx.left(0).is_none(), "top-left left");
        // Mid-tile lookups also `None` (no blocks decoded yet).
        assert!(ctx.above(3).is_none());
        assert!(ctx.left(5).is_none());
    }

    /// Out-of-range `mi_col` and `mi_row_in_sb` return `None`
    /// without panicking — defensive accessor behaviour matching
    /// libvpx's "the dummy row/col only spans the real mi grid"
    /// invariant.
    #[test]
    fn out_of_range_returns_none() {
        let ctx = TileDecodeCtx::new(4);
        assert!(ctx.above(4).is_none()); // == tile_mi_cols
        assert!(ctx.above(99).is_none());
        assert!(ctx.left(8).is_none()); // == MI_BLOCK_SIZE
        assert!(ctx.left(99).is_none());
    }

    /// After writing an 8×8 block at `(0, 0)`, the block directly
    /// to its right reads it as `left`; the block directly below
    /// (next SB row) reads it as `above`. Other slots stay `None`.
    /// This is the "frame interior" / "tile interior" case.
    #[test]
    fn write_8x8_block_at_origin() {
        let mut ctx = TileDecodeCtx::new(8);
        let mi = mk_intra(TxSize::Tx4x4, false, 0);
        ctx.write_block(0, 0, 1, 1, &mi);

        // The block to the right (mi_col=1, mi_row_in_sb=0) sees
        // `mi` as its left neighbour and `None` as its above
        // (since we're still on the topmost SB row).
        assert_eq!(ctx.left(0), Some(&mi));
        assert!(ctx.above(0).is_some(), "above[0] populated by bottom-row projection");
        assert_eq!(ctx.above(0), Some(&mi));
        // Adjacent columns / rows untouched.
        assert!(ctx.above(1).is_none());
        assert!(ctx.left(1).is_none());
    }

    /// Frame-edge (mi_row=0, mi_col=0) case: even after decoding
    /// the corner block, a block at `(mi_row=0, mi_col=0)` itself
    /// would read `(None, None)`. This is the "first block of the
    /// frame" case and it's exactly what
    /// [`rvp9_core::mode_info::get_tx_size_context`] expects to
    /// collapse to `max_tx_size` on both sides.
    #[test]
    fn frame_corner_reads_none_none() {
        // Before any writes, the corner block sees no neighbours.
        let ctx = TileDecodeCtx::new(8);
        assert!(ctx.above(0).is_none());
        assert!(ctx.left(0).is_none());

        // Use the integration with `get_tx_size_context` to confirm
        // the contract. The result for `(None, None, max=Tx32x32)`
        // is 1 (libvpx `(2*max) > max` always true).
        let above = ctx.above(0).map(|m| (m.tx_size, m.skip));
        let left = ctx.left(0).map(|m| (m.tx_size, m.skip));
        let ctx_index = rvp9_core::mode_info::get_tx_size_context(
            above,
            left,
            TxSize::Tx32x32,
        );
        assert_eq!(ctx_index, 1, "frame-corner tx_size context");
    }

    /// Frame-edge (top edge): a block in the first SB row but not
    /// the first column. After writing the left neighbour, the
    /// block sees `left = Some(...)` but `above = None`. Mirrors
    /// libvpx's `up_available = 0, left_available = 1` case.
    #[test]
    fn top_edge_reads_left_only() {
        let mut ctx = TileDecodeCtx::new(8);
        let left_mi = mk_intra(TxSize::Tx4x4, false, 0);
        ctx.write_block(0, 0, 1, 1, &left_mi);
        // Block at (mi_row_in_sb=0, mi_col=1): above[1] hasn't been
        // written to (no block in column 1 yet), left[0] has.
        assert!(ctx.above(1).is_none(), "top-edge: above neighbour absent");
        assert_eq!(ctx.left(0), Some(&left_mi));
    }

    /// Frame-edge (left edge): a block in the first column but not
    /// the first SB row. After writing the above neighbour (from
    /// the previous SB row) and resetting `left`, the block sees
    /// `above = Some(...)` but `left = None`. Mirrors libvpx's
    /// `up_available = 1, left_available = 0` case.
    #[test]
    fn left_edge_reads_above_only() {
        let mut ctx = TileDecodeCtx::new(8);
        let above_mi = mk_intra(TxSize::Tx8x8, false, 0);
        // Pretend we decoded a block at column 0 in the previous
        // SB row — it left a trail in `above[0]`.
        ctx.write_block(7, 0, 1, 1, &above_mi);
        // Now we move to the next SB row: reset left.
        ctx.reset_left();
        // Block at (mi_row_in_sb=0, mi_col=0): above[0] from the
        // previous SB row's last-row, left[0] cleared.
        assert_eq!(ctx.above(0), Some(&above_mi));
        assert!(ctx.left(0).is_none(), "left-edge: left neighbour absent");
    }

    /// SB-boundary case: after `reset_left()`, all left slots are
    /// `None` again but `above` slots persist (they're filled by
    /// the previous SB row's bottom edges).
    #[test]
    fn reset_left_clears_only_left() {
        let mut ctx = TileDecodeCtx::new(8);
        let mi_a = mk_intra(TxSize::Tx16x16, true, 5);
        let mi_b = mk_intra(TxSize::Tx8x8, false, 2);
        // mi_a at (mi_row=0, mi_col=3): writes above[3] and left[0].
        ctx.write_block(0, 3, 1, 1, &mi_a);
        // mi_b at (mi_row=2, mi_col=0): writes above[0] and left[2].
        ctx.write_block(2, 0, 1, 1, &mi_b);
        assert_eq!(ctx.above(3), Some(&mi_a));
        assert_eq!(ctx.above(0), Some(&mi_b));
        assert_eq!(ctx.left(0), Some(&mi_a));
        assert_eq!(ctx.left(2), Some(&mi_b));

        ctx.reset_left();
        // Above survives.
        assert_eq!(ctx.above(3), Some(&mi_a));
        assert_eq!(ctx.above(0), Some(&mi_b));
        // Left cleared.
        assert!(ctx.left(0).is_none());
        assert!(ctx.left(2).is_none());
    }

    /// Multi-row / multi-column block (16×16, n_8x8=2×2): the
    /// write splatters across 2 above slots and 2 left slots.
    #[test]
    fn write_16x16_block_splatters_2x2() {
        let mut ctx = TileDecodeCtx::new(8);
        let mi = mk_intra_16x16(TxSize::Tx16x16, false);
        // 16×16 block at (mi_row_in_sb=2, mi_col=4).
        ctx.write_block(2, 4, 2, 2, &mi);

        // Above row [4..6] populated; outside not touched.
        assert!(ctx.above(3).is_none());
        assert_eq!(ctx.above(4), Some(&mi));
        assert_eq!(ctx.above(5), Some(&mi));
        assert!(ctx.above(6).is_none());

        // Left column [2..4] populated; outside not touched.
        assert!(ctx.left(1).is_none());
        assert_eq!(ctx.left(2), Some(&mi));
        assert_eq!(ctx.left(3), Some(&mi));
        assert!(ctx.left(4).is_none());
    }

    /// Multi-row / single-column block (16×8, n_8x8=2×1): two
    /// `above` slots, one `left` slot.
    #[test]
    fn write_16x8_block_splatter() {
        let mut ctx = TileDecodeCtx::new(8);
        let mi = mk_inter_16x8();
        ctx.write_block(3, 1, 2, 1, &mi);

        assert!(ctx.above(0).is_none());
        assert_eq!(ctx.above(1), Some(&mi));
        assert_eq!(ctx.above(2), Some(&mi));
        assert!(ctx.above(3).is_none());

        assert!(ctx.left(2).is_none());
        assert_eq!(ctx.left(3), Some(&mi));
        assert!(ctx.left(4).is_none());
    }

    /// Writes that exceed `tile_mi_cols` get clamped — no panic,
    /// out-of-bounds bytes simply don't get written. Mirrors
    /// libvpx's `x_mis = MIN(bw, mi_cols - mi_col)` clamp.
    #[test]
    fn write_clamped_at_tile_right_edge() {
        let mut ctx = TileDecodeCtx::new(4);
        let mi = mk_intra_16x16(TxSize::Tx16x16, false);
        // 16×16 block straddling the right edge: starts at col=3,
        // would write cols 3 and 4 but tile only has cols 0..4.
        ctx.write_block(0, 3, 2, 2, &mi);
        assert!(ctx.above(2).is_none());
        assert_eq!(ctx.above(3), Some(&mi));
        // col 4 doesn't exist; just verify no panic + accessor returns None.
        assert!(ctx.above(4).is_none());
    }

    /// Writes whose height exceeds `MI_BLOCK_SIZE = 8` get clamped
    /// — the `left` column only covers the current SB row.
    #[test]
    fn write_clamped_at_sb_bottom() {
        let mut ctx = TileDecodeCtx::new(8);
        let mi = mk_intra(TxSize::Tx4x4, false, 0);
        // Pretend a block tries to write rows 6..10 (impossible in
        // practice — max block height is 8 mi). Verify clamping.
        ctx.write_block(6, 0, 1, 4, &mi);
        assert!(ctx.left(5).is_none());
        assert_eq!(ctx.left(6), Some(&mi));
        assert_eq!(ctx.left(7), Some(&mi));
        // index 8 is out of bounds.
        assert!(ctx.left(8).is_none());
    }

    /// Subsequent writes overwrite earlier ones in the same slot.
    /// This is the libvpx invariant: the **last** block to occupy
    /// a slot is what the next neighbour-reading block sees.
    #[test]
    fn later_writes_overwrite_earlier() {
        let mut ctx = TileDecodeCtx::new(8);
        let mi_a = mk_intra(TxSize::Tx4x4, false, 1);
        let mi_b = mk_intra(TxSize::Tx16x16, true, 7);
        ctx.write_block(0, 2, 1, 1, &mi_a);
        assert_eq!(ctx.above(2), Some(&mi_a));
        ctx.write_block(0, 2, 1, 1, &mi_b);
        assert_eq!(ctx.above(2), Some(&mi_b));
        assert_eq!(ctx.left(0), Some(&mi_b));
    }

    /// `new(0)` panics. Tiles always span at least one MI column.
    #[test]
    #[should_panic(expected = "tile must span at least one")]
    fn new_zero_panics() {
        let _ = TileDecodeCtx::new(0);
    }

    /// Integration: walk a few neighbour reads through
    /// `get_tx_size_context` to confirm the ctx feeds the
    /// downstream consumer correctly. Hand-traced expected values.
    #[test]
    fn integration_with_get_tx_size_context() {
        use rvp9_core::mode_info::get_tx_size_context;

        let mut ctx = TileDecodeCtx::new(8);
        // Decode a Tx16x16 / not-skipped block at (0, 0).
        let mi_corner = mk_intra(TxSize::Tx16x16, false, 0);
        ctx.write_block(0, 0, 1, 1, &mi_corner);

        // Block at (0, 1): above=None (top edge), left=Some(Tx16x16, false).
        let above = ctx.above(1).map(|m| (m.tx_size, m.skip));
        let left = ctx.left(0).map(|m| (m.tx_size, m.skip));
        // max=Tx32x32: above mirrors left → 2+2=4 > 3 → ctx 1.
        let c = get_tx_size_context(above, left, TxSize::Tx32x32);
        assert_eq!(c, 1, "top-edge with left=Tx16x16 → ctx 1");

        // Decode another Tx4x4 / skipped block at (0, 1).
        let mi_right = mk_intra(TxSize::Tx4x4, true, 0);
        ctx.write_block(0, 1, 1, 1, &mi_right);
        // Block at (1, 0): above=Some(Tx4x4, true), left=None (SB
        // wasn't reset; we're still in row 0 of the SB). Wait —
        // (1, 0) is in the SAME SB row, so left[1] is the left
        // neighbour of (mi_row_in_sb=1, mi_col=0). But we want to
        // check the "second SB row, first column" case: simulate
        // that by resetting left.
        ctx.reset_left();
        // Now block at (mi_row_in_sb=0, mi_col=1) on second SB row:
        // above=Some(Tx4x4, true) (from `mi_right`), left=None.
        let above = ctx.above(1).map(|m| (m.tx_size, m.skip));
        let left = ctx.left(0).map(|m| (m.tx_size, m.skip));
        // above is Tx4x4 + skipped → contributes max=3. left
        // missing → mirrors above. Sum = 3+3=6 > 3 → ctx 1.
        let c = get_tx_size_context(above, left, TxSize::Tx32x32);
        assert_eq!(c, 1, "left-edge with above=Tx4x4-skipped → ctx 1");

        // Block at (mi_row_in_sb=0, mi_col=0) on second SB row:
        // above=Some(Tx16x16, false) (from corner block writing
        // above[0]), left=None.
        let above = ctx.above(0).map(|m| (m.tx_size, m.skip));
        let left = ctx.left(0).map(|m| (m.tx_size, m.skip));
        // above contributes 2 (not skipped). left mirrors → 2+2=4
        // > 3 → ctx 1.
        let c = get_tx_size_context(above, left, TxSize::Tx32x32);
        assert_eq!(c, 1, "left-edge with above=Tx16x16 → ctx 1");
    }
}
