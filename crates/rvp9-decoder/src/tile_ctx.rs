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
//! We mirror libvpx directly: a full 2-D `mi_grid` — one
//! [`ModeInfo`] per 8×8 slot in the tile, row-major with stride
//! `tile_mi_cols`. Each per-block decode reads its neighbours
//! straight from the grid:
//!
//! - **`above(mi_row, mi_col)`** = `grid[mi_row - 1][mi_col]`
//!   (libvpx `above_mi = xd->mi - mi_stride`).
//! - **`left(mi_row, mi_col)`** = `grid[mi_row][mi_col - 1]`
//!   (libvpx `left_mi = xd->mi - 1`).
//!
//! Because VP9 decodes in raster order, the above `(r-1, c)` and left
//! `(r, c-1)` cells are always written before the current block reads
//! them (or are out of the tile), so — unlike libvpx's per-SB-row
//! `left_context` — no reset is needed; the grid is the canonical
//! state. The grid also feeds `find_mv_refs`, which needs arbitrary
//! `(±row, ±col)` neighbours, via [`TileDecodeCtx::mi_at_relative`].
//!
//! An earlier design kept a separate above-row + left-column
//! `Vec<Option<ModeInfo>>` reduction *alongside* the grid; it was a
//! redundant per-block copy of the same data and was removed (the
//! accessors now read the grid directly).
//!
//! ## Frame-edge / tile-edge semantics
//!
//! Neighbour reads return `None` at the tile top (`mi_row == 0`) and
//! tile left edge (`mi_col == 0`) — the signed underflow / bounds
//! guard in [`TileDecodeCtx::mi_at_relative`] — matching libvpx
//! `up_available = 0` / `left_available = 0`. The `(None, None)` arm
//! in [`rvp9_core::mode_info::get_tx_size_context`] collapses both
//! sides to `max_tx_size`, equivalent to libvpx's
//! `xd->above_mi == NULL && xd->left_mi == NULL` path.
//!
//! ## Usage
//!
//! The per-block decode loop instantiates one [`TileDecodeCtx`] per
//! tile (via [`TileDecodeCtx::new_with_view`]), walks SBs row-by-row,
//! and at each 8×8 block:
//!
//! ```ignore
//! let above = ctx.above(mi_row, mi_col);
//! let left  = ctx.left(mi_row, mi_col);
//! let tx_ctx = get_tx_size_context(
//!     above.map(|mi| (mi.tx_size, mi.skip)),
//!     left .map(|mi| (mi.tx_size, mi.skip)),
//!     max_tx_size,
//! );
//! // ... decode block ...
//! ctx.write_block(mi_row, mi_col, n_8x8_wide, n_8x8_high, &decoded_mi);
//! ```

use rvp9_core::inter::mv_ref::{NeighbourLookup, Position};
use rvp9_core::mode_info::ModeInfo;

use crate::tile::TileBounds;

// Re-export the coefficient-context types so callers still doing
// `use crate::tile_ctx::{TileCoefCtx, combine_entropy_contexts, MAX_MB_PLANE}`
// continue to work. The canonical home is now
// `rvp9_core::coef_ctx` (promoted there so the encoder can share it);
// these re-exports preserve the long-standing decoder-side import
// path documented in `detokenize.rs`, `reconstruct.rs`,
// `reconstruct_inter.rs`, and `tile.rs`.
pub use rvp9_core::coef_ctx::{MAX_MB_PLANE, TileCoefCtx, combine_entropy_contexts};


/// Non-overlapping mutable view into a rectangular sub-region of the
/// frame-wide MI grid. Analogous to `TileFrameView` for pixels.
///
/// # Safety invariants (construction-time)
///
/// 1. The parent `Vec<Option<ModeInfo>>` must outlive all `TileMiView`s.
/// 2. No two `TileMiView`s may cover overlapping cell ranges.
/// 3. No other `&mut` to the parent Vec exists while any view is alive.
#[allow(dead_code)]
pub struct TileMiView {
    /// Raw pointer to the first cell of the frame-wide grid (row 0, col 0).
    ptr: *mut Option<ModeInfo>,
    /// Total number of cells in the grid (mi_rows * mi_cols).
    len: usize,
    /// Frame-wide stride (= mi_cols).
    mi_stride: usize,
    /// Tile's column bounds (inclusive start, exclusive end).
    mi_col_start: usize,
    mi_col_end: usize,
    /// Tile's row bounds (inclusive start, exclusive end).
    mi_row_start: usize,
    mi_row_end: usize,
}

// SAFETY: TileMiView is Send because:
// 1. Raw pointer into a single Vec allocation.
// 2. Construction contract guarantees non-overlapping write ranges;
//    each tile writes only within [mi_row_start..mi_row_end,
//    mi_col_start..mi_col_end].
// 3. The parent Vec's lifetime is guaranteed by the scoped thread pool.
unsafe impl Send for TileMiView {}

impl TileMiView {
    /// Construct a `TileMiView` covering the tile described by `bounds`.
    ///
    /// # Safety
    ///
    /// Caller must guarantee:
    /// - `ptr` points to the first cell of a `Vec<Option<ModeInfo>>`
    ///   with at least `len` elements.
    /// - The Vec outlives this view.
    /// - No two views constructed from the same Vec overlap in their
    ///   `(mi_row_start..mi_row_end, mi_col_start..mi_col_end)` rectangles.
    /// - No other `&mut` to the Vec exists while any view is alive.
    pub unsafe fn new(
        ptr: *mut Option<ModeInfo>,
        len: usize,
        mi_stride: usize,
        bounds: &TileBounds,
    ) -> Self {
        Self {
            ptr,
            len,
            mi_stride,
            mi_col_start: bounds.mi_col_start,
            mi_col_end: bounds.mi_col_end,
            mi_row_start: bounds.mi_row_start,
            mi_row_end: bounds.mi_row_end,
        }
    }
}

/// Per-tile neighbour-context state.
///
/// Carries the 2-D `mi_grid` (libvpx `mi_grid_visible`-equivalent,
/// `vp9_alloccommon.c:vp9_alloc_context_buffers`) the per-block decoder
/// reads to compute its prediction contexts (the above / left
/// neighbours) and that [`NeighbourLookup`] callers scan for arbitrary
/// `Position { row, col }` offsets — e.g. `(-1, 6)`, the top-right of a
/// 64×64 SB — in `find_mv_refs`. See the module docs for the rationale.
///
/// One instance per tile. The grid is `tile_mi_rows × tile_mi_cols`,
/// addressed by tile-relative `(mi_row, mi_col)` coordinates, row-major
/// with `mi_stride = tile_mi_cols`. Each cell is `Option<ModeInfo>`:
/// `None` for tile-edge / not-yet-decoded positions, `Some(mi)` for
/// cells covered by a decoded block. libvpx replicates the parent's
/// `MODE_INFO*` across every covered 8×8 cell so accessors at any
/// sub-position return the same parent — we mirror that splat.
///
/// In production the grid is a [`TileMiView`] into the frame-wide grid
/// (via [`Self::new_with_view`]); [`Self::new_with_grid`] allocates a
/// tile-local grid (tests); [`Self::new`] allocates no grid (a
/// degenerate legacy/test path — every neighbour read returns `None`).
pub struct TileDecodeCtx {
    /// Tile width in mi-units (8-px columns), i.e.
    /// `tile.mi_col_end - tile.mi_col_start`.
    ///
    /// The above/left neighbour lookups read the 2D grid DIRECTLY
    /// (`above(r,c) = grid[r-1][c]`, `left(r,c) = grid[r][c-1]`, with
    /// tile-edge `None` semantics) — exactly libvpx's `above_mi =
    /// xd->mi - mi_stride` / `left_mi = xd->mi - 1` pointer arithmetic
    /// over `mi_grid_visible`. The former per-block above-row +
    /// left-column `Vec<Option<ModeInfo>>` reductions were redundant
    /// caches of that same grid data (a full-struct copy per block per
    /// projected cell) and were removed.
    tile_mi_cols: usize,
    /// 2D mi grid, row-major `tile_mi_rows * tile_mi_cols`. Empty
    /// when this context was constructed via [`Self::new`] (legacy
    /// paths that don't need the grid). See struct docs.
    mi_grid: Vec<Option<ModeInfo>>,
    /// Number of rows in the 2D grid. `0` when no grid was
    /// allocated. Equals `mi_row_end - mi_row_start` for the tile.
    tile_mi_rows: usize,
    /// Phase 2: optional mutable view into the pre-allocated frame-wide
    /// MI grid. `None` when using the legacy `new`/`new_with_grid` path.
    mi_view: Option<TileMiView>,
    /// Phase 2: frame-absolute column offset.
    mi_col_start: usize,
    /// Phase 2: frame-absolute row offset.
    mi_row_start: usize,
}

impl TileDecodeCtx {
    /// Create a degenerate per-tile context with NO mi grid.
    ///
    /// `tile_mi_cols` is the tile's width in mi-units (8-px units),
    /// i.e. `tile.mi_col_end - tile.mi_col_start`. Without a grid,
    /// [`Self::above`] / [`Self::left`] / [`Self::mi_at`] /
    /// [`Self::mi_at_relative`] always return `None` and
    /// [`Self::write_block`] is a no-op — this is a legacy/test-only
    /// path. Production uses [`Self::new_with_view`]; tests needing a
    /// grid use [`Self::new_with_grid`].
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
            tile_mi_cols,
            mi_grid: Vec::new(),
            tile_mi_rows: 0,
            mi_view: None,
            mi_col_start: 0,
            mi_row_start: 0,
        }
    }

    /// Create a fresh per-tile context with the full 2D mi grid
    /// allocated (all cells `None`).
    ///
    /// `tile_mi_cols` / `tile_mi_rows` are the tile's dimensions in
    /// mi-units (8-px units), i.e.
    /// `tile.mi_col_end - tile.mi_col_start` /
    /// `tile.mi_row_end - tile.mi_row_start`. All grid cells start
    /// `None` (libvpx's `mi_grid_visible` post-`vpx_calloc` state).
    ///
    /// # Panics
    ///
    /// If `tile_mi_cols == 0` or `tile_mi_rows == 0` — every legal
    /// VP9 tile spans at least one SB row and column.
    #[must_use]
    pub fn new_with_grid(tile_mi_cols: usize, tile_mi_rows: usize) -> Self {
        assert!(
            tile_mi_cols > 0,
            "tile must span at least one 8×8 mode-info column"
        );
        assert!(
            tile_mi_rows > 0,
            "tile must span at least one 8×8 mode-info row"
        );
        Self {
            tile_mi_cols,
            mi_grid: vec![None; tile_mi_cols * tile_mi_rows],
            tile_mi_rows,
            mi_view: None,
            mi_col_start: 0,
            mi_row_start: 0,
        }
    }

    /// Create a fresh per-tile context with a [`TileMiView`] into
    /// the pre-allocated frame-wide MI grid (Phase 2).
    ///
    /// `mi_view` provides raw-pointer access to the frame-wide grid.
    /// `mi_col_start` / `mi_row_start` are the frame-absolute offsets.
    #[must_use]
    pub fn new_with_view(
        tile_mi_cols: usize,
        tile_mi_rows: usize,
        mi_view: TileMiView,
        mi_col_start: usize,
        mi_row_start: usize,
    ) -> Self {
        assert!(
            tile_mi_cols > 0,
            "tile must span at least one 8\u{00d7}8 mode-info column"
        );
        assert!(
            tile_mi_rows > 0,
            "tile must span at least one 8\u{00d7}8 mode-info row"
        );
        Self {
            tile_mi_cols,
            mi_grid: Vec::new(),
            tile_mi_rows,
            mi_view: Some(mi_view),
            mi_col_start,
            mi_row_start,
        }
    }

    /// Number of 8×8 columns covered by this tile.
    #[must_use]
    pub fn tile_mi_cols(&self) -> usize {
        self.tile_mi_cols
    }

    /// Number of 8×8 rows covered by this tile's 2D mi grid. Returns
    /// `0` when the context was constructed via [`Self::new`] (no
    /// grid allocated).
    #[must_use]
    pub fn tile_mi_rows(&self) -> usize {
        self.tile_mi_rows
    }

    /// Borrow the raw 2D mi grid (row-major, stride =
    /// [`Self::tile_mi_cols`]). Empty slice when the context was
    /// constructed via [`Self::new`] (no grid allocated).
    ///
    /// Mirrors libvpx's `mi_grid_visible` access for callers that need
    /// to walk the whole tile (e.g. K2.c's loop-filter mask builder).
    #[must_use]
    pub fn mi_grid(&self) -> &[Option<ModeInfo>] {
        &self.mi_grid
    }

    /// Read the [`ModeInfo`] at absolute tile-relative
    /// `(mi_row, mi_col)`. Returns `None` if the position is outside
    /// the tile (frame top / bottom / left / right edge equivalent)
    /// or has not yet been written by a decoded block. Also returns
    /// `None` when the context was constructed without a grid (legacy
    /// 1-D paths via [`Self::new`]).
    ///
    /// This mirrors libvpx `xd->mi[row * mi_stride + col]` indexing
    /// (`vp9_onyxc_int.h:mi_grid_visible`), gated by libvpx's
    /// `is_inside(tile, ...)` predicate
    /// (`vp9_mvref_common.h:is_inside`).
    #[must_use]
    pub fn mi_at(&self, mi_row: usize, mi_col: usize) -> Option<&ModeInfo> {
        if mi_row >= self.tile_mi_rows || mi_col >= self.tile_mi_cols {
            return None;
        }
        if let Some(ref view) = self.mi_view {
            // Phase 2: read from the frame-wide grid at absolute coords.
            let abs_row = self.mi_row_start + mi_row;
            let abs_col = self.mi_col_start + mi_col;
            let idx = abs_row * view.mi_stride + abs_col;
            debug_assert!(idx < view.len, "mi_at: index {idx} out of grid bounds {}", view.len);
            // SAFETY: idx is within [mi_row_start..mi_row_end,
            // mi_col_start..mi_col_end] which is within the grid allocation.
            unsafe { (*view.ptr.add(idx)).as_ref() }
        } else {
            let stride = self.tile_mi_cols;
            self.mi_grid.get(mi_row * stride + mi_col)?.as_ref()
        }
    }

    /// Read the [`ModeInfo`] at `(base_row + delta_row,
    /// base_col + delta_col)`. Returns `None` if the resulting
    /// position is outside the tile (negative or beyond
    /// `tile_mi_rows` / `tile_mi_cols`), or if the cell has not yet
    /// been written, or if no grid was allocated.
    ///
    /// Signed arithmetic: the libvpx `find_mv_refs` neighbour table
    /// (`vp9_mvref_common.h:mv_ref_blocks`) includes positions like
    /// `Position { row: -1, col: 6 }` (top-right of a 64×64 SB) and
    /// `Position { row: 4, col: -1 }` (bottom-left of a 32×64 SB)
    /// — both of which translate to signed mi-row / mi-col offsets.
    #[must_use]
    pub fn mi_at_relative(
        &self,
        base_row: i32,
        base_col: i32,
        delta_row: i32,
        delta_col: i32,
    ) -> Option<&ModeInfo> {
        let row = base_row.checked_add(delta_row)?;
        let col = base_col.checked_add(delta_col)?;
        if row < 0 || col < 0 {
            return None;
        }
        self.mi_at(row as usize, col as usize)
    }

    /// Read the above neighbour of the block at tile-relative
    /// `(mi_row, mi_col)` — the grid cell directly above it,
    /// `(mi_row - 1, mi_col)`. Returns `None` at the tile top
    /// (frame/tile edge — libvpx `above_mi == NULL` / `up_available =
    /// 0`). Exactly libvpx's `above_mi = xd->mi - mi_stride` read over
    /// `mi_grid_visible`, gated by the tile-edge bounds in [`mi_at`].
    /// The above neighbour at `(mi_row-1, mi_col)` is always decoded
    /// before the current block in raster order (or out of bounds), so
    /// the read never observes a stale (cross-frame persistent) cell.
    #[must_use]
    pub fn above(&self, mi_row: usize, mi_col: usize) -> Option<&ModeInfo> {
        self.mi_at_relative(mi_row as i32, mi_col as i32, -1, 0)
    }

    /// Read the left neighbour of the block at tile-relative
    /// `(mi_row, mi_col)` — the grid cell directly to its left,
    /// `(mi_row, mi_col - 1)`. Returns `None` at the tile's left edge
    /// (libvpx `left_mi == NULL` / `left_available = 0`). Exactly
    /// libvpx's `left_mi = xd->mi - 1` read, gated by the tile-edge
    /// bounds. The left neighbour at `(mi_row, mi_col-1)` is decoded
    /// before the current block in raster order (or out of bounds), so
    /// the read never observes a stale cell.
    #[must_use]
    pub fn left(&self, mi_row: usize, mi_col: usize) -> Option<&ModeInfo> {
        self.mi_at_relative(mi_row as i32, mi_col as i32, 0, -1)
    }

    /// Splat a decoded block's [`ModeInfo`] into every covered cell of
    /// the 2D mi grid, so later blocks read it back as their above /
    /// left neighbour (via [`above`](Self::above) / [`left`](Self::left))
    /// and `find_mv_refs` reads it via
    /// [`mi_at_relative`](Self::mi_at_relative). Mirrors libvpx's
    /// `set_offsets` writing `xd->mi[...] = mi` for every covered slot in
    /// `mi_grid_visible` (`vp9_decodeframe.c:set_offsets`).
    ///
    /// Parameters:
    /// - `mi_row`, `mi_col`: top-left of the block, in mi-units, relative
    ///   to the tile (`0..tile_mi_rows` / `0..tile_mi_cols`).
    /// - `n_8x8_wide`, `n_8x8_high`: block size in mi-units
    ///   (`bsize.n_8x8_wide()` / `n_8x8_high()`); sub-8 partitions clamp
    ///   both to 1, so a 4×4 block writes its single covering cell.
    /// - `mi`: the decoded [`ModeInfo`] deposited into every covered cell.
    ///
    /// Writes are clamped to the tile/grid bounds; out-of-bounds cells are
    /// dropped (libvpx's `x_mis`/`y_mis` clamp, `vp9_decodeframe.c:920`).
    /// On the [`Self::new`] (no-grid) path this is a no-op.
    pub fn write_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        n_8x8_wide: usize,
        n_8x8_high: usize,
        mi: &ModeInfo,
    ) {
        // Hoist the `Some(*mi)` copy out of the splat loop: bind it once so
        // each covered grid cell is a `slice.fill` of a local rather than
        // re-dereferencing `mi` and re-constructing the `Option` per
        // iteration. Byte-identical value written to the identical cells.
        let mi_val = Some(*mi);

        // 2D grid splat: write `mi` into every covered cell. The
        // `above` / `left` neighbour lookups read these cells back from the
        // grid directly — there is no separate above-row / left-column
        // reduction to keep in sync (it was a redundant per-block copy of
        // this same data). Writes into either the frame-wide grid via
        // TileMiView (Phase 2) or the tile-local mi_grid (legacy).
        if let Some(ref view) = self.mi_view {
            // Phase 2: frame-absolute coordinates via TileMiView.
            let abs_col = self.mi_col_start + mi_col;
            let abs_row = self.mi_row_start + mi_row;
            let stride = view.mi_stride;
            let row_end_grid = (abs_row + n_8x8_high).min(view.mi_row_end);
            let col_end_grid = (abs_col + n_8x8_wide).min(view.mi_col_end);
            for r in abs_row..row_end_grid {
                let base = r * stride;
                // SAFETY: we only write within [mi_row_start..mi_row_end,
                // mi_col_start..mi_col_end] -- guaranteed non-overlapping
                // across tiles by VP9 spec (tile boundaries are SB-aligned).
                unsafe {
                    let slice = std::slice::from_raw_parts_mut(
                        view.ptr.add(base + abs_col),
                        col_end_grid - abs_col,
                    );
                    slice.fill(mi_val);
                }
            }
        } else if !self.mi_grid.is_empty() {
            // Legacy path: tile-local grid.
            let stride = self.tile_mi_cols;
            let row_end_grid = (mi_row + n_8x8_high).min(self.tile_mi_rows);
            let col_end_grid = (mi_col + n_8x8_wide).min(stride);
            for r in mi_row..row_end_grid {
                let base = r * stride;
                self.mi_grid[base + mi_col..base + col_end_grid].fill(mi_val);
            }
        }
    }
}

/// Bridge adapter that lets a [`TileDecodeCtx`]'s 2D mi grid satisfy
/// [`rvp9_core::inter::mv_ref::NeighbourLookup`]. The trait's
/// `get(pos)` callback receives positions relative to a per-block
/// origin; the wrapper captures that origin (`base_mi_row` /
/// `base_mi_col`, in tile-relative mi-units) and forwards through
/// [`TileDecodeCtx::mi_at_relative`].
///
/// Constructed per-block at the start of the inter-mode neighbour
/// scan (Phase D wireup). The wrapper holds a borrow of the tile
/// context, so the underlying `mi_grid` cannot be mutated for the
/// lifetime of the wrapper — neighbour lookups always read the
/// already-decoded predecessors.
///
/// ## Position semantics
///
/// `Position { row, col }` from
/// [`rvp9_core::inter::mv_ref::MV_REF_BLOCKS`] is interpreted as a
/// signed offset in 8×8 MI units from the current block's top-left
/// corner. `Position { row: -1, col: 0 }` is "directly above";
/// `Position { row: -1, col: 6 }` is "top-right of a 64×64 SB" —
/// both reach into the already-decoded area and resolve through
/// [`TileDecodeCtx::mi_at_relative`] which clamps to tile bounds.
pub struct TileNeighbourLookup<'a> {
    ctx: &'a TileDecodeCtx,
    base_mi_row: i32,
    base_mi_col: i32,
}

impl<'a> TileNeighbourLookup<'a> {
    /// Construct a wrapper anchored at the given block's top-left
    /// `(mi_row, mi_col)` (tile-relative, in 8×8 MI units).
    #[must_use]
    pub fn new(ctx: &'a TileDecodeCtx, base_mi_row: i32, base_mi_col: i32) -> Self {
        Self {
            ctx,
            base_mi_row,
            base_mi_col,
        }
    }
}

impl NeighbourLookup for TileNeighbourLookup<'_> {
    fn get(&self, pos: Position) -> Option<&ModeInfo> {
        self.ctx.mi_at_relative(
            self.base_mi_row,
            self.base_mi_col,
            i32::from(pos.row),
            i32::from(pos.col),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::block::BlockSize;
    use rvp9_core::intra_pred::IntraMode;
    use rvp9_core::mode_info::BlockMode;
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

    /// Fresh grid: no block decoded yet, so every above/left neighbour
    /// lookup returns `None` (libvpx `above_mi == NULL` / `left_mi ==
    /// NULL` at the frame/tile top-left).
    #[test]
    fn fresh_grid_neighbours_none() {
        let ctx = TileDecodeCtx::new_with_grid(8, 8);
        assert_eq!(ctx.tile_mi_cols(), 8);
        assert!(ctx.above(0, 0).is_none(), "top-left above");
        assert!(ctx.left(0, 0).is_none(), "top-left left");
        // Mid-grid lookups also `None` (no blocks decoded yet).
        assert!(ctx.above(3, 3).is_none());
        assert!(ctx.left(5, 5).is_none());
    }

    /// Out-of-range `(mi_row, mi_col)` return `None` without panicking
    /// (the `mi_at` bounds guard), matching libvpx's "the dummy row/col
    /// only spans the real mi grid" invariant.
    #[test]
    fn out_of_range_returns_none() {
        let ctx = TileDecodeCtx::new_with_grid(4, 4);
        // above(0, 4): col == tile_mi_cols → grid(-1, 4) out of range.
        assert!(ctx.above(0, 4).is_none());
        assert!(ctx.above(99, 99).is_none());
        assert!(ctx.left(4, 0).is_none()); // row == tile_mi_rows
        assert!(ctx.left(99, 99).is_none());
    }

    /// After writing an 8×8 block at grid `(0, 0)`, the block directly
    /// below it `(1, 0)` reads it as its `above` neighbour and the
    /// block directly to its right `(0, 1)` reads it as its `left`
    /// neighbour — exactly libvpx `above_mi = mi - mi_stride` /
    /// `left_mi = mi - 1`. The written block itself reads `None` on
    /// both edges (tile top-left).
    #[test]
    fn above_left_read_grid_neighbours() {
        let mut ctx = TileDecodeCtx::new_with_grid(8, 8);
        let mi = mk_intra(TxSize::Tx4x4, false, 0);
        ctx.write_block(0, 0, 1, 1, &mi);

        assert_eq!(ctx.above(1, 0), Some(&mi), "block below reads it as above");
        assert_eq!(ctx.left(0, 1), Some(&mi), "block right reads it as left");
        assert!(ctx.above(0, 0).is_none(), "tile top: no above");
        assert!(ctx.left(0, 0).is_none(), "tile left: no left");
        // Neighbours pointing at undecoded cells stay `None`.
        assert!(ctx.above(1, 1).is_none(), "grid(0,1) undecoded");
        assert!(ctx.left(1, 1).is_none(), "grid(1,0) undecoded");
    }

    /// Tile edges: a block on the top row has no `above`; a block in
    /// the left column has no `left` — even when those cells are
    /// decoded — because the lookup steps off the tile via signed
    /// underflow in `mi_at_relative`. Also exercises multi-cell
    /// `write_block` splat (16×16 = 2×2 mi).
    #[test]
    fn tile_edge_neighbours_none() {
        let mut ctx = TileDecodeCtx::new_with_grid(8, 8);
        let mi = mk_intra_16x16(TxSize::Tx16x16, false);
        // Cover the whole grid with 16×16 (2×2 mi) blocks.
        for r in (0..8).step_by(2) {
            for c in (0..8).step_by(2) {
                ctx.write_block(r, c, 2, 2, &mi);
            }
        }
        // Interior neighbour reads hit decoded cells.
        assert_eq!(ctx.above(2, 0), Some(&mi));
        assert_eq!(ctx.left(0, 2), Some(&mi));
        // Top row: no above. Left column: no left.
        for c in 0..8 {
            assert!(ctx.above(0, c).is_none(), "above at top row col {c}");
        }
        for r in 0..8 {
            assert!(ctx.left(r, 0).is_none(), "left at left col row {r}");
        }
    }

    /// Later writes overwrite earlier ones in a covered cell: the
    /// **last** block to occupy a grid cell is what the next
    /// neighbour-reading block sees (libvpx `set_offsets` invariant).
    #[test]
    fn later_writes_overwrite_earlier() {
        let mut ctx = TileDecodeCtx::new_with_grid(8, 8);
        let mi_a = mk_intra(TxSize::Tx4x4, false, 1);
        let mi_b = mk_intra(TxSize::Tx16x16, true, 7);
        ctx.write_block(0, 2, 1, 1, &mi_a);
        assert_eq!(ctx.above(1, 2), Some(&mi_a));
        ctx.write_block(0, 2, 1, 1, &mi_b);
        assert_eq!(ctx.above(1, 2), Some(&mi_b));
        assert_eq!(ctx.left(0, 3), Some(&mi_b));
    }

    /// `new(0)` panics. Tiles always span at least one MI column.
    #[test]
    #[should_panic(expected = "tile must span at least one")]
    fn new_zero_panics() {
        let _ = TileDecodeCtx::new(0);
    }

    /// Integration: neighbour reads feed `get_tx_size_context`
    /// correctly through the grid. Hand-traced expected values.
    #[test]
    fn integration_with_get_tx_size_context() {
        use rvp9_core::mode_info::get_tx_size_context;

        let mut ctx = TileDecodeCtx::new_with_grid(8, 8);
        // Tx16x16 / not-skipped block at grid (0, 0).
        let mi_corner = mk_intra(TxSize::Tx16x16, false, 0);
        ctx.write_block(0, 0, 1, 1, &mi_corner);

        // Block at (0, 1): above = grid(-1, 1) = None (top edge);
        // left = grid(0, 0) = Some(Tx16x16, false). max=Tx32x32:
        // above mirrors left → 2+2=4 > 3 → ctx 1.
        let above = ctx.above(0, 1).map(|m| (m.tx_size, m.skip));
        let left = ctx.left(0, 1).map(|m| (m.tx_size, m.skip));
        assert_eq!(
            get_tx_size_context(above, left, TxSize::Tx32x32),
            1,
            "top-edge with left=Tx16x16 → ctx 1",
        );

        // Block at (1, 0): above = grid(0, 0) = Some(Tx16x16, false);
        // left = grid(1, -1) = None (left edge). 2+2=4 > 3 → ctx 1.
        let above = ctx.above(1, 0).map(|m| (m.tx_size, m.skip));
        let left = ctx.left(1, 0).map(|m| (m.tx_size, m.skip));
        assert_eq!(
            get_tx_size_context(above, left, TxSize::Tx32x32),
            1,
            "left-edge with above=Tx16x16 → ctx 1",
        );
    }

    // ===================================================================
    // 2D mi grid tests (Stage 3 Phase B.3)
    // ===================================================================

    /// Construct a grid, write a single 8×8 block at (0, 0), read it
    /// back via `mi_at(0, 0)`. The adjacent cell (0, 1) stays `None`
    /// — the splat covers only the block's `n_8x8 × n_8x8` region.
    #[test]
    fn mi_grid_write_single_8x8_at_origin() {
        let mut ctx = TileDecodeCtx::new_with_grid(8, 8);
        let mi = mk_intra(TxSize::Tx4x4, false, 0);
        ctx.write_block(0, 0, 1, 1, &mi);

        assert_eq!(ctx.mi_at(0, 0), Some(&mi));
        assert!(ctx.mi_at(0, 1).is_none(), "adjacent column untouched");
        assert!(ctx.mi_at(1, 0).is_none(), "adjacent row untouched");
    }

    /// 16×16 block at (4, 4): the splat covers a 2×2 region of mi
    /// cells; every cell returns the same parent ModeInfo. Mirrors
    /// libvpx's `mi_grid_visible[r * mi_stride + c] = mi` over the
    /// whole `x_mis × y_mis` region.
    #[test]
    fn mi_grid_16x16_splats_across_2x2() {
        let mut ctx = TileDecodeCtx::new_with_grid(8, 8);
        let mi = mk_intra_16x16(TxSize::Tx16x16, false);
        ctx.write_block(4, 4, 2, 2, &mi);

        assert_eq!(ctx.mi_at(4, 4), Some(&mi));
        assert_eq!(ctx.mi_at(4, 5), Some(&mi));
        assert_eq!(ctx.mi_at(5, 4), Some(&mi));
        assert_eq!(ctx.mi_at(5, 5), Some(&mi));
        // The cells just outside the splat region stay None.
        assert!(ctx.mi_at(3, 4).is_none());
        assert!(ctx.mi_at(6, 4).is_none());
        assert!(ctx.mi_at(4, 3).is_none());
        assert!(ctx.mi_at(4, 6).is_none());
    }

    /// `mi_at_relative` with negative deltas: reading at base=(0, 0)
    /// with delta=(-1, 0) is "directly above the top-left block" —
    /// must return None (out of tile via signed-int arithmetic).
    /// Mirrors libvpx's `is_inside` failure for `mv_ref->row + mi_row
    /// < 0`.
    #[test]
    fn mi_at_relative_negative_returns_none() {
        let ctx = TileDecodeCtx::new_with_grid(8, 8);
        assert!(ctx.mi_at_relative(0, 0, -1, 0).is_none());
        assert!(ctx.mi_at_relative(0, 0, 0, -1).is_none());
        assert!(ctx.mi_at_relative(0, 0, -1, -1).is_none());
        // mi_at with usize-cast equivalents of negative positions
        // also returns None (we test the i32-path here; the
        // accessor doesn't accept signed coords, so this is the
        // only path callers can use).
    }

    /// Reading past the tile bottom / right returns None. Combined
    /// with the negative-delta test this covers all four tile-edge
    /// cases.
    #[test]
    fn mi_at_returns_none_outside_tile() {
        let ctx = TileDecodeCtx::new_with_grid(4, 4);
        assert!(ctx.mi_at(4, 0).is_none(), "row == tile_mi_rows");
        assert!(ctx.mi_at(0, 4).is_none(), "col == tile_mi_cols");
        assert!(ctx.mi_at(99, 99).is_none(), "far out of range");
    }

    /// libvpx `mv_ref_blocks[Block64x64][7] = (-1, 6)`: a 64×64 SB
    /// scans its top-right neighbour at row -1, col base+6. From the
    /// SB origin (mi_row=0), that resolves to row=-1 which is above
    /// the tile → must return None. Pins B.2's gap that this 2D
    /// grid closes.
    #[test]
    fn mi_at_relative_top_right_of_64x64_returns_none_at_top() {
        let ctx = TileDecodeCtx::new_with_grid(16, 16);
        // SB top-left at (0, 0); `Position { row: -1, col: 6 }`.
        assert!(ctx.mi_at_relative(0, 0, -1, 6).is_none());
        // From row >= 1 the same delta resolves to a valid row but
        // an unwritten cell — still None until the block is written.
        assert!(ctx.mi_at_relative(8, 0, -1, 6).is_none());
    }

    /// `mi_at_relative` does the addition: base=(5, 10) +
    /// delta=(-1, 0) reads (4, 10). Confirms the math is the simple
    /// signed addition libvpx uses, not e.g. coordinate scaling.
    #[test]
    fn mi_at_relative_math_pins_base_plus_delta() {
        let mut ctx = TileDecodeCtx::new_with_grid(16, 16);
        let mi = mk_intra(TxSize::Tx8x8, false, 0);
        // Write a block at (4, 10).
        ctx.write_block(4, 10, 1, 1, &mi);
        // base=(5, 10) + delta=(-1, 0) → row=4, col=10.
        assert_eq!(ctx.mi_at_relative(5, 10, -1, 0), Some(&mi));
        // base=(5, 10) + delta=(0, -1) → row=5, col=9 (unwritten).
        assert!(ctx.mi_at_relative(5, 10, 0, -1).is_none());
    }

    /// `TileDecodeCtx::new` (no grid) is a degenerate legacy/test
    /// constructor: `write_block` is a no-op and every neighbour lookup
    /// (`above` / `left` / `mi_at`) returns `None`, since there is no
    /// grid to read or write. Production always uses `new_with_view` /
    /// `new_with_grid`.
    #[test]
    fn no_grid_ctx_returns_none_everywhere() {
        let mut ctx = TileDecodeCtx::new(8);
        let mi = mk_intra(TxSize::Tx4x4, false, 0);
        ctx.write_block(0, 0, 1, 1, &mi); // no-op without a grid
        assert!(ctx.above(1, 0).is_none());
        assert!(ctx.left(0, 1).is_none());
        assert!(ctx.mi_at(0, 0).is_none());
        assert!(ctx.mi_at_relative(1, 0, -1, 0).is_none());
        assert_eq!(ctx.tile_mi_rows(), 0);
    }

    /// NeighbourLookup bridge: drive `find_mv_refs` through the
    /// `TileNeighbourLookup` adapter. The 64×64 candidate-walk
    /// includes `(-1, 6)` (top-right) — from the tile origin that
    /// resolves to None and the walk completes without panic. This
    /// pins the wedge that B.2 flagged.
    #[test]
    fn neighbour_lookup_bridge_handles_out_of_tile_positions() {
        use rvp9_core::block::BlockSize;
        use rvp9_core::inter::mv_ref::{find_mv_refs, BlockBounds};
        use rvp9_core::mode_info::{InterMode, RefFrame};

        let ctx = TileDecodeCtx::new_with_grid(16, 16);
        let lookup = TileNeighbourLookup::new(&ctx, 0, 0);
        let bounds = BlockBounds {
            left: 0,
            right: 1024,
            top: 0,
            bottom: 1024,
        };
        let sign_bias = [false; 4];

        // Empty tile, every neighbour position resolves to None
        // (including the `(-1, 6)` top-right scan for 64×64).
        // No panic, returns the zero-filled list.
        let list = find_mv_refs(
            InterMode::NearMv,
            BlockSize::Block64x64,
            RefFrame::LastFrame,
            &lookup,
            &sign_bias,
            &bounds,
            None,
        );
        // NEARMV → count = 2 with zeros (the loop falls through).
        assert_eq!(list.count, 2);
    }

    /// NeighbourLookup bridge: a real `(-1, 0)` (directly above)
    /// neighbour written into the grid is returned to
    /// `find_mv_refs`, which folds its MV into the candidate list.
    /// Confirms the bridge actually passes ModeInfo through, not
    /// just None.
    #[test]
    fn neighbour_lookup_bridge_returns_written_block() {
        use rvp9_core::bitstream::InterpFilter;
        use rvp9_core::block::BlockSize;
        use rvp9_core::inter::mv_ref::{find_mv_refs, BlockBounds};
        use rvp9_core::mode_info::{BlockMode, InterMode, Mv, RefFrame};

        let mut ctx = TileDecodeCtx::new_with_grid(16, 16);
        let mv = Mv { row: 24, col: -16 };
        let above = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NewMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [mv, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        // Above neighbour at (mi_row=0, mi_col=0..2) — a 16×16 block.
        ctx.write_block(0, 0, 2, 2, &above);

        // Current block at (mi_row=2, mi_col=0); `(-1, 0)` resolves
        // to (mi_row=1, mi_col=0), which is INSIDE the 16×16 splat.
        let lookup = TileNeighbourLookup::new(&ctx, 2, 0);
        let bounds = BlockBounds {
            left: 0,
            right: 1024,
            top: 0,
            bottom: 1024,
        };
        let sign_bias = [false; 4];
        let list = find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &lookup,
            &sign_bias,
            &bounds,
            None,
        );
        assert_eq!(list.mvs[0], mv, "above neighbour's MV propagated through bridge");
        assert_eq!(list.count, 1);
    }

    /// Cross-grid splat is clamped at the tile right / bottom edge.
    /// A 32×32 block at (mi_row=6, mi_col=6) in an 8×8 tile would
    /// cover (6..10, 6..10), but only (6..8, 6..8) lies inside the
    /// tile. No panic; the in-tile cells are populated.
    #[test]
    fn mi_grid_splat_clamps_at_tile_edges() {
        let mut ctx = TileDecodeCtx::new_with_grid(8, 8);
        let mi = mk_intra_16x16(TxSize::Tx16x16, false);
        // Pretend a 32×32 block at (mi_row=6, mi_col=6): would write
        // (6..10) × (6..10). Tile is 8×8, so only (6..8) × (6..8)
        // fit.
        ctx.write_block(6, 6, 4, 4, &mi);
        for r in 6..8 {
            for c in 6..8 {
                assert_eq!(ctx.mi_at(r, c), Some(&mi), "in-tile cell ({r}, {c})");
            }
        }
        // No panic for the out-of-range readback.
        assert!(ctx.mi_at(8, 8).is_none());
    }

    // ===================================================================
    // TileCoefCtx tests (Stage 3 Task 8.2.7, Wave 3 F.A)
    // ===================================================================

    /// `combine_entropy_contexts` pins the libvpx formula
    /// (`vp9/common/vp9_entropy.h:161-164`):
    ///     return (a != 0) + (b != 0);
    /// Hand-traced expected values for every (a, l) ∈ {0, 1}².
    /// We also probe with the "any non-zero" values libvpx actually
    /// writes (always `0` or `1` from `(eob > 0)`) to confirm the
    /// `u8` → `bool` reduction matches.
    #[test]
    fn combine_entropy_contexts_matches_libvpx() {
        assert_eq!(combine_entropy_contexts(0, 0), 0);
        assert_eq!(combine_entropy_contexts(0, 1), 1);
        assert_eq!(combine_entropy_contexts(1, 0), 1);
        assert_eq!(combine_entropy_contexts(1, 1), 2);
        // Any non-zero ⇒ counts as 1, regardless of value (libvpx
        // `ENTROPY_CONTEXT` is `char` but only ever holds 0/1 from
        // `decode_coefs`'s `(eob > 0)` writeback — we still cover
        // the defensive path).
        assert_eq!(combine_entropy_contexts(5, 0), 1);
        assert_eq!(combine_entropy_contexts(0, 200), 1);
        assert_eq!(combine_entropy_contexts(7, 3), 2);
    }

    /// Fresh `TileCoefCtx`: every cell starts zero, every accessor
    /// returns 0. This is libvpx's post-`vpx_calloc` /
    /// post-`memset(..., 0, ...)` state.
    #[test]
    fn fresh_coef_ctx_is_all_zero() {
        let ctx = TileCoefCtx::new(8);
        // Y above row covers 8 * 2 = 16 cells.
        assert_eq!(ctx.above_coef_width(0), 16);
        // UV above row halves under 4:2:0.
        assert_eq!(ctx.above_coef_width(1), 8);
        assert_eq!(ctx.above_coef_width(2), 8);
        // Y left col covers 16 cells; UV halves to 8.
        assert_eq!(ctx.left_coef_height(0), 16);
        assert_eq!(ctx.left_coef_height(1), 8);
        assert_eq!(ctx.left_coef_height(2), 8);
        // Every cell starts zero.
        for plane in 0..MAX_MB_PLANE {
            for x4 in 0..ctx.above_coef_width(plane) {
                assert_eq!(ctx.above_coef(plane, x4), 0);
            }
            for y4 in 0..ctx.left_coef_height(plane) {
                assert_eq!(ctx.left_coef(plane, y4), 0);
            }
        }
        // The initial token-tree context at the frame corner is 0.
        let pt = combine_entropy_contexts(ctx.above_coef(0, 0), ctx.left_coef(0, 0));
        assert_eq!(pt, 0, "frame corner pt = 0");
    }

    /// `set_coef_flags` writes the correct cells and the readback
    /// matches. Hand-traced: a TX_8X8 block at (x4=4, y4=2) in plane
    /// 0 with `nonzero=true` should set above[4..6] = 1 and
    /// left[2..4] = 1; everything else stays 0.
    #[test]
    fn set_coef_flags_tx8x8_writes_2x2_region() {
        let mut ctx = TileCoefCtx::new(8);
        ctx.set_coef_flags(0, 4, 2, 2, 2, true);

        // Y above row: only cells 4 and 5 written.
        for x4 in 0..16 {
            let expected = if (4..6).contains(&x4) { 1 } else { 0 };
            assert_eq!(
                ctx.above_coef(0, x4),
                expected,
                "Y above[{x4}] expected {expected}"
            );
        }
        // Y left col: only cells 2 and 3 written.
        for y4 in 0..16 {
            let expected = if (2..4).contains(&y4) { 1 } else { 0 };
            assert_eq!(
                ctx.left_coef(0, y4),
                expected,
                "Y left[{y4}] expected {expected}"
            );
        }
        // U / V planes untouched.
        for plane in 1..3 {
            for x4 in 0..ctx.above_coef_width(plane) {
                assert_eq!(ctx.above_coef(plane, x4), 0);
            }
            for y4 in 0..ctx.left_coef_height(plane) {
                assert_eq!(ctx.left_coef(plane, y4), 0);
            }
        }
    }

    /// `set_coef_flags(..., nonzero=false)` clears cells back to
    /// zero. Mirrors the libvpx `(eob > 0) == false` writeback path:
    /// the block had no non-zero coefficients, so subsequent
    /// neighbours see "no contribution" from it.
    #[test]
    fn set_coef_flags_zero_clears() {
        let mut ctx = TileCoefCtx::new(4);
        // First, mark cells as non-zero.
        ctx.set_coef_flags(0, 0, 0, 4, 4, true);
        for i in 0..4 {
            assert_eq!(ctx.above_coef(0, i), 1);
            assert_eq!(ctx.left_coef(0, i), 1);
        }
        // Then clear with nonzero=false.
        ctx.set_coef_flags(0, 0, 0, 4, 4, false);
        for i in 0..4 {
            assert_eq!(ctx.above_coef(0, i), 0);
            assert_eq!(ctx.left_coef(0, i), 0);
        }
    }

    /// `combine_entropy_contexts` driven through the full pipeline:
    /// write some flags, read above + left of a neighbouring block,
    /// confirm `pt` matches the hand-traced expectation. Pins the
    /// integration that Wave 3 F.B (decode_coefs) consumes.
    #[test]
    fn coef_ctx_drives_combine_entropy_contexts() {
        let mut ctx = TileCoefCtx::new(8);
        // Block A at Y x4=2, y4=0, TX_4X4, nonzero=true → writes
        // above[2] = 1 and left[0] = 1.
        ctx.set_coef_flags(0, 2, 0, 1, 1, true);
        // Block B (to the right of A): x4=3, y4=0. Reads above[3]
        // (= 0, A didn't reach it) and left[0] (= 1, A wrote it).
        let pt = combine_entropy_contexts(ctx.above_coef(0, 3), ctx.left_coef(0, 0));
        assert_eq!(pt, 1, "block right of A: pt = 0 + 1 = 1");
        // Block C (directly below A's column in next SB row): reads
        // above[2] = 1, left[0] = 0 (after reset_left_coef).
        ctx.reset_left_coef();
        let pt = combine_entropy_contexts(ctx.above_coef(0, 2), ctx.left_coef(0, 0));
        assert_eq!(pt, 1, "block below A: pt = 1 + 0 = 1");
        // Now set both above and left for some test position.
        ctx.set_coef_flags(0, 5, 3, 1, 1, true);
        // Read at the same cell: above[5] = 1, left[3] = 1.
        let pt = combine_entropy_contexts(ctx.above_coef(0, 5), ctx.left_coef(0, 3));
        assert_eq!(pt, 2, "both-non-zero: pt = 2");
    }

    /// SB-row boundary: `reset_left_coef` clears every plane's left
    /// column but `above_coef` survives. Mirrors libvpx
    /// `vp9_decodeframe.c:2089` (between-SB-row reset) while
    /// `cm->above_context` is only re-zeroed at frame start
    /// (`vp9_decodeframe.c:2049-2052`).
    #[test]
    fn reset_left_coef_preserves_above_coef() {
        let mut ctx = TileCoefCtx::new(8);
        // Populate every plane.
        ctx.set_coef_flags(0, 0, 0, 4, 4, true); // Y
        ctx.set_coef_flags(1, 0, 0, 2, 2, true); // U
        ctx.set_coef_flags(2, 0, 0, 2, 2, true); // V

        // Pre-reset: both above and left populated.
        assert_eq!(ctx.above_coef(0, 1), 1);
        assert_eq!(ctx.left_coef(0, 1), 1);
        assert_eq!(ctx.above_coef(1, 1), 1);
        assert_eq!(ctx.left_coef(1, 1), 1);

        ctx.reset_left_coef();

        // Above persists.
        assert_eq!(ctx.above_coef(0, 1), 1, "Y above survives reset");
        assert_eq!(ctx.above_coef(1, 1), 1, "U above survives reset");
        assert_eq!(ctx.above_coef(2, 1), 1, "V above survives reset");
        // Left cleared on every plane.
        for plane in 0..MAX_MB_PLANE {
            for y4 in 0..ctx.left_coef_height(plane) {
                assert_eq!(
                    ctx.left_coef(plane, y4),
                    0,
                    "plane {plane} left[{y4}] cleared"
                );
            }
        }
    }

    /// 4:2:0 chroma subsampling pin: UV planes' above-row is
    /// exactly half the width of Y's, and UV planes' left-col is
    /// half the height of Y's. Confirms the Profile-0-only
    /// hardcoding in `TileCoefCtx::new`.
    #[test]
    fn uv_dimensions_are_half_of_y_under_4_2_0() {
        for tile_mi_cols in [1, 2, 5, 8, 32] {
            let ctx = TileCoefCtx::new(tile_mi_cols);
            // Y above width = tile_mi_cols * 2.
            assert_eq!(ctx.above_coef_width(0), tile_mi_cols * 2);
            // U / V above width = half of Y.
            assert_eq!(ctx.above_coef_width(1), tile_mi_cols);
            assert_eq!(ctx.above_coef_width(2), tile_mi_cols);
            // Y left height = 8 mi (one SB) * 2 = 16.
            assert_eq!(ctx.left_coef_height(0), 16);
            // U / V left height = 8.
            assert_eq!(ctx.left_coef_height(1), 8);
            assert_eq!(ctx.left_coef_height(2), 8);
        }
    }

    /// Out-of-range accesses return 0 without panicking. Mirrors
    /// libvpx's "the dummy row/col only spans the real mi grid"
    /// invariant — anything outside is treated as "no contribution".
    #[test]
    fn out_of_range_coef_access_returns_zero() {
        let ctx = TileCoefCtx::new(4);
        // Plane index out of range.
        assert_eq!(ctx.above_coef(3, 0), 0);
        assert_eq!(ctx.above_coef(99, 0), 0);
        assert_eq!(ctx.left_coef(3, 0), 0);
        // Cell index out of range.
        assert_eq!(ctx.above_coef(0, 8), 0); // Y above width = 8
        assert_eq!(ctx.above_coef(0, 999), 0);
        assert_eq!(ctx.left_coef(0, 16), 0); // Y left height = 16
        assert_eq!(ctx.left_coef(1, 8), 0); // U left height = 8
    }

    /// `set_coef_flags` clamps writes at the tile-right edge and
    /// SB-bottom edge — no panic, out-of-range cells are silently
    /// dropped. Mirrors libvpx's `get_ctx_shift` (`vp9_detokenize.c:
    /// 263-273`) for the edge-clamp case.
    #[test]
    fn set_coef_flags_clamps_at_edges() {
        let mut ctx = TileCoefCtx::new(4); // Y above width = 8
        // TX_32X32 (8×8 4×4 sub-blocks) at the right edge of the
        // tile: starts at x4=4, would write x4=4..12 but only 4..8
        // are in range.
        ctx.set_coef_flags(0, 4, 0, 8, 8, true);
        for x4 in 0..4 {
            assert_eq!(ctx.above_coef(0, x4), 0, "left of write region");
        }
        for x4 in 4..8 {
            assert_eq!(ctx.above_coef(0, x4), 1, "in write region");
        }
        // No panic at x4=8 (out of range) — returns 0.
        assert_eq!(ctx.above_coef(0, 8), 0);

        // Left clamping: TX_32X32 at y4=12 in the SB. y4=12..20
        // requested, only 12..16 fit. (Note this also re-writes
        // above[0..8] — clamped no-op since the above row only
        // has 8 cells.)
        ctx.set_coef_flags(0, 0, 12, 8, 8, true);
        for y4 in 12..16 {
            assert_eq!(ctx.left_coef(0, y4), 1, "in left write region");
        }
        // y4=16 out of range.
        assert_eq!(ctx.left_coef(0, 16), 0);
    }

    /// `TileCoefCtx::new(0)` panics. Tiles always span ≥ 1 MI col.
    #[test]
    #[should_panic(expected = "tile must span at least one")]
    fn new_zero_tile_panics() {
        let _ = TileCoefCtx::new(0);
    }
}
