//! Per-tile coefficient-context plumbing: the per-plane `above` row and
//! `left` column of `ENTROPY_CONTEXT` bytes that the token-coefficient
//! decoder (and the matching encoder path) reads/writes around every
//! transform block.
//!
//! ## Scope
//!
//! Originally landed in `rvp9-decoder` for Stage 3 Task 8.2.7 (Wave 3
//! F.A). Promoted to `rvp9-core` so the encoder can share the same
//! structure for its symmetric coefficient-encoding path — the data
//! layout, reset semantics, and accessors are identical on both sides.
//!
//! See [`TileCoefCtx`] for the storage model and reset rules, and
//! [`combine_entropy_contexts`] for the per-block initial-context
//! computation libvpx performs at the top of `decode_coefs` (and at the
//! matching point in `vp9_encode_block_tokens`).

/// libvpx `MI_BLOCK_SIZE` (`vp9_enums.h:25`). Each 64×64 superblock
/// covers an 8×8 mode-info grid.
const MI_BLOCK_SIZE: usize = 8;

/// Number of 4×4 sub-blocks per 8×8 mi-unit along one axis.
/// libvpx `MI_SIZE = 8` px ÷ 4 px = 2 (`vp9_enums.h:24`).
const SUB4X4_PER_MI: usize = 2;

/// Number of luma planes + chroma planes the coef-context tables
/// span. libvpx `MAX_MB_PLANE = 3` (Y, U, V).
/// See `vp9/common/vp9_blockd.h:32`.
pub const MAX_MB_PLANE: usize = 3;

/// Per-plane chroma subsampling for VP9 Profile 0 (4:2:0).
/// Plane 0 (Y) is full-resolution; planes 1/2 (U/V) are
/// half-resolution in both axes. We hardcode this because Profile 0
/// is the only path Stage 3 ships in v1 (master ledger D-01 RESOLVED).
const PROFILE0_SUBSAMPLING_X: [usize; MAX_MB_PLANE] = [0, 1, 1];
const PROFILE0_SUBSAMPLING_Y: [usize; MAX_MB_PLANE] = [0, 1, 1];

/// libvpx `combine_entropy_contexts` (`vp9/common/vp9_entropy.h:161-164`).
///
/// Computes the initial token-tree context for `decode_coefs` by
/// counting how many of the (above, left) coef-context bytes are
/// non-zero. The result is in `0..=2`:
///
/// - `0`: both neighbours coded all-zeros (or are unavailable).
/// - `1`: exactly one neighbour had a non-zero coefficient.
/// - `2`: both neighbours had non-zero coefficients.
///
/// libvpx writes this as `(a != 0) + (b != 0)` against signed
/// `ENTROPY_CONTEXT` bytes. Our `u8` representation is equivalent:
/// the only values written are 0 and 1 (from `(eob > 0)` in
/// `vp9_decode_block_tokens` at `vp9_detokenize.c:294/302/311/323`).
///
/// Wave 3 F.B (`decode_coefs`) calls this once at block entry to
/// derive the `pt` argument the token-tree walk consumes.
#[must_use]
#[inline]
pub fn combine_entropy_contexts(a: u8, l: u8) -> usize {
    usize::from(a != 0) + usize::from(l != 0)
}

/// Per-tile coefficient-context state.
///
/// Mirrors libvpx's per-plane `above_context[plane]` row + per-plane
/// `left_context[plane]` column of `ENTROPY_CONTEXT` bytes used by
/// `vp9_decode_block_tokens` (`vp9/common/vp9_blockd.h:191-192`,
/// allocation at `vp9/common/vp9_alloccommon.c:138-149`). The
/// values written are `(eob > 0)` — i.e. "did any non-zero
/// coefficient land in the corresponding 4×4 sub-block".
///
/// ## Scope (Stage 3 Task 8.2.7, "Wave 3 F.A")
///
/// Wave 3 F.B (`decode_coefs` core) consumes these arrays to derive
/// the initial token-tree context per transform block via
/// [`combine_entropy_contexts`]. After decoding a transform block it
/// writes the resulting "had any non-zero" flag back into the
/// covered sub-block cells (mirrors `vp9_detokenize.c:294/302/311/
/// 323`'s `a[..] = l[..] = (eob > 0)` splats).
///
/// ## Storage model — VP9 Profile 0 only
///
/// libvpx stores three independent per-plane rows in one packed
/// buffer (`vp9_alloccommon.c:138`: `2 * mi_cols_aligned_to_sb *
/// MAX_MB_PLANE`). We instead carry one `Vec<u8>` per plane sized
/// per-plane:
///
/// - **`above_coef[0]` (Y)** — `tile_mi_cols * 2` cells (2 4×4
///   sub-blocks per 8×8 mi-unit).
/// - **`above_coef[1]`, `above_coef[2]` (U, V)** — `tile_mi_cols`
///   cells (half-width under 4:2:0).
/// - **`left_coef[0]` (Y)** — `MI_BLOCK_SIZE * 2 = 16` cells (one
///   per 4×4 row in the current SB row).
/// - **`left_coef[1]`, `left_coef[2]` (U, V)** — `8` cells (half-
///   height under 4:2:0).
///
/// Profile 0 is the only path Stage 3 ships in v1 — the master
/// ledger pins D-01 RESOLVED on this. If we ever extend to Profile
/// 1/2/3 we'll thread real subsampling factors through `new`.
///
/// ## Reset semantics
///
/// - **`above_coef`** persists across SB rows within a tile and is
///   reset only at frame start (libvpx `vp9_decodeframe.c:2049-
///   2052` `memset(cm->above_context, 0, …)` before tile-row loop).
/// - **`left_coef`** resets at the start of every SB row (libvpx
///   `vp9_decodeframe.c:2089` `vp9_zero(tile_data->xd.left_context)`).
///
/// ## Initial state
///
/// All cells start `0`. This matches libvpx's `vpx_calloc` plus the
/// `memset(..., 0, ...)` at the top of `decode_tiles`. The frame-
/// edge / tile-edge case for [`combine_entropy_contexts`] therefore
/// collapses naturally: an unwritten cell reads as `0`, which the
/// combiner treats as "no non-zero coefficient in that neighbour".
#[derive(Clone, Debug)]
pub struct TileCoefCtx {
    /// Per-plane above-row coef-context bytes. One slot per 4×4
    /// sub-block column across the tile width. Persists across SB
    /// rows within the tile.
    above_coef: [Vec<u8>; MAX_MB_PLANE],
    /// Per-plane left-column coef-context bytes. One slot per 4×4
    /// sub-block row within the currently-active SB row. Reset to
    /// zero at the start of each new SB row via
    /// [`TileCoefCtx::reset_left_coef`].
    left_coef: [Vec<u8>; MAX_MB_PLANE],
}

impl TileCoefCtx {
    /// Construct a fresh per-tile coef context for VP9 Profile 0
    /// (8-bit 4:2:0). `tile_mi_cols` is the tile's width measured
    /// in mi-units (`tile.mi_col_end - tile.mi_col_start`).
    ///
    /// Per-plane width is `tile_mi_cols * 2 / (1 << subsampling_x)`
    /// 4×4 cells; per-plane SB-row height is `MI_BLOCK_SIZE * 2 /
    /// (1 << subsampling_y) = 16 / 8 / 8` cells.
    ///
    /// All cells start zero — matches libvpx `vpx_calloc` +
    /// the per-frame `memset(..., 0, ...)` at
    /// `vp9_decodeframe.c:2049-2052`.
    ///
    /// # Panics
    ///
    /// If `tile_mi_cols == 0` — every legal VP9 tile spans at least
    /// one SB column.
    #[must_use]
    pub fn new(tile_mi_cols: usize) -> Self {
        assert!(
            tile_mi_cols > 0,
            "tile must span at least one 8×8 mode-info column"
        );
        let y_above_w = tile_mi_cols * SUB4X4_PER_MI;
        let y_left_h = MI_BLOCK_SIZE * SUB4X4_PER_MI;
        let above_coef = [
            vec![0u8; y_above_w >> PROFILE0_SUBSAMPLING_X[0]],
            vec![0u8; y_above_w >> PROFILE0_SUBSAMPLING_X[1]],
            vec![0u8; y_above_w >> PROFILE0_SUBSAMPLING_X[2]],
        ];
        let left_coef = [
            vec![0u8; y_left_h >> PROFILE0_SUBSAMPLING_Y[0]],
            vec![0u8; y_left_h >> PROFILE0_SUBSAMPLING_Y[1]],
            vec![0u8; y_left_h >> PROFILE0_SUBSAMPLING_Y[2]],
        ];
        Self { above_coef, left_coef }
    }

    /// Read the above-row coef-context byte for plane `plane` at
    /// 4×4-sub-block column `x4`. Returns `0` at frame top (no
    /// block has been written into that column yet — libvpx
    /// `above_context` cell still at its `calloc`/`memset` zero)
    /// and for any out-of-range index (defensive — callers
    /// shouldn't request these).
    ///
    /// `plane` is `0`/`1`/`2` for `Y`/`U`/`V`; `x4` is the 4×4
    /// column index within the tile (NOT the mi-unit index — one
    /// mi-unit covers 2 4×4 cells for Y, 1 for UV at 4:2:0).
    #[must_use]
    pub fn above_coef(&self, plane: usize, x4: usize) -> u8 {
        self.above_coef
            .get(plane)
            .and_then(|row| row.get(x4).copied())
            .unwrap_or(0)
    }

    /// Read the left-column coef-context byte for plane `plane` at
    /// 4×4-sub-block row `y4_in_sb` (within the current SB row).
    /// Returns `0` at the tile's left edge (libvpx
    /// `left_context` cell still zero after `vp9_zero`) and for
    /// any out-of-range index.
    ///
    /// `y4_in_sb` is the 4×4 row within the current SB (`0..16` for
    /// Y; `0..8` for UV at 4:2:0).
    #[must_use]
    pub fn left_coef(&self, plane: usize, y4_in_sb: usize) -> u8 {
        self.left_coef
            .get(plane)
            .and_then(|col| col.get(y4_in_sb).copied())
            .unwrap_or(0)
    }

    /// Reset every `left_coef` plane to zero at the start of a new
    /// SB row. Mirrors libvpx `vp9_decodeframe.c:2089`
    /// `vp9_zero(tile_data->xd.left_context)` between SB-row
    /// iterations.
    pub fn reset_left_coef(&mut self) {
        for col in &mut self.left_coef {
            for slot in col {
                *slot = 0;
            }
        }
    }

    /// Splat the post-`decode_coefs` "had any non-zero coefficient"
    /// flag across the (w4 × h4) 4×4 sub-block region just decoded
    /// in plane `plane`.
    ///
    /// Mirrors the writeback in libvpx `vp9_detokenize.c:294/302/
    /// 311/323` — every covered above/left cell receives the same
    /// `(eob > 0)` value.
    ///
    /// Parameters:
    /// - `plane`: `0`/`1`/`2` for Y/U/V.
    /// - `x4_start`: top-left 4×4 column in the tile (NOT mi-unit).
    /// - `y4_in_sb_start`: top-left 4×4 row within the current SB
    ///   row (NOT mi-unit).
    /// - `w4`, `h4`: width and height of the transform block in
    ///   4×4 sub-blocks. For `TX_4X4` both are `1`; for `TX_8X8`
    ///   both are `2`; for `TX_16X16` both are `4`; for `TX_32X32`
    ///   both are `8`.
    /// - `nonzero`: the `(eob > 0)` flag — `true` ⇒ writes `1`,
    ///   `false` ⇒ writes `0`.
    ///
    /// Writes are clamped to the plane's array bounds; out-of-
    /// bounds cells are silently dropped. Mirrors libvpx's
    /// `get_ctx_shift` clamping path at the right / bottom edges
    /// (`vp9_detokenize.c:263-273`) — we don't bother computing the
    /// shifted-byte trick because our per-cell representation
    /// handles partial coverage natively.
    pub fn set_coef_flags(
        &mut self,
        plane: usize,
        x4_start: usize,
        y4_in_sb_start: usize,
        w4: usize,
        h4: usize,
        nonzero: bool,
    ) {
        let v = u8::from(nonzero);
        if let Some(above_row) = self.above_coef.get_mut(plane) {
            let end = (x4_start + w4).min(above_row.len());
            if x4_start < end {
                for slot in &mut above_row[x4_start..end] {
                    *slot = v;
                }
            }
        }
        if let Some(left_col) = self.left_coef.get_mut(plane) {
            let end = (y4_in_sb_start + h4).min(left_col.len());
            if y4_in_sb_start < end {
                for slot in &mut left_col[y4_in_sb_start..end] {
                    *slot = v;
                }
            }
        }
    }

    /// Width of the above-row table for plane `plane`, in 4×4
    /// sub-block units. Returns `0` for out-of-range `plane`.
    /// Useful for callers that need to bound block-write loops.
    #[must_use]
    pub fn above_coef_width(&self, plane: usize) -> usize {
        self.above_coef.get(plane).map_or(0, Vec::len)
    }

    /// Height of the left-column table for plane `plane`, in 4×4
    /// sub-block units. Returns `0` for out-of-range `plane`.
    #[must_use]
    pub fn left_coef_height(&self, plane: usize) -> usize {
        self.left_coef.get(plane).map_or(0, Vec::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            // Y left height = MI_BLOCK_SIZE * 2 = 16.
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
