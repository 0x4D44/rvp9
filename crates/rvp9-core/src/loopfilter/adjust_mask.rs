//! Post-pass on a 64×64 [`LoopFilterMask`] that performs frame-edge
//! cleanup after [`super::setup_mask::setup_mask`] (Task 6.3.5).
//!
//! Mirrors libvpx `vp9_adjust_mask` (`vp9_loopfilter.c:766-878`).
//!
//! Three responsibilities, run in this order:
//!
//! 1. **TX_32X32 → TX_16X16 merge.** No 32-tap loop filter exists
//!    (the kernel palette is 4/8/16-tap), so any bit set for
//!    `TX_32X32` gets OR'd into the `TX_16X16` slot. The
//!    `TX_32X32` slots are **not** cleared — libvpx leaves them as
//!    they were, since the apply pass only reads `TX_16X16` and
//!    below.
//! 2. **4×4-border → 8×8 promotion.** Any `TX_4X4` bit that sits on
//!    a per-8×8-column "left edge" (luma columns 0, 8, 16, 24, …
//!    in the 64-bit mask) or per-8×8-row "above edge" (luma rows
//!    0, 8, 16, 24, …) gets promoted to the `TX_8X8` slot, then
//!    cleared from `TX_4X4`. This guarantees SB-internal 8×8
//!    boundaries always get at least an 8-tap filter even when the
//!    underlying block uses TX_4×4.
//! 3. **Frame-edge masking.** When the superblock straddles the
//!    right or bottom frame edge, the bits for past-frame 8×8
//!    cells get cleared. The very-first column (mi_col == 0) also
//!    gets its left-edge bit cleared, because the implicit
//!    frame-border itself is never filtered. There is no `mi_row
//!    == 0` clear here — libvpx handles the top-row case inside
//!    the per-plane filter pass (`vp9_loopfilter.c:1252-1259`),
//!    not in adjust_mask. See LLD §5.5.
//!
//! After steps 2 and 3 the libvpx asserts at `:862-877` hold: no
//! 8×8 cell has more than one TX-size bit set for left_y / above_y
//! / left_uv / above_uv. These show up as `debug_assert!` here.

use super::mask::LoopFilterMask;
use super::mask_tables::{ABOVE_BORDER_UV, ABOVE_BORDER_Y, LEFT_BORDER_UV, LEFT_BORDER_Y};

/// MI-grid side length of a 64×64 superblock (in 8×8 units).
/// Mirrors libvpx `MI_BLOCK_SIZE` (`vp9_enums.h:25`).
const MI_BLOCK_SIZE: u32 = 8;

/// Index of `TX_8X8` in the per-TX-size mask arrays. Matches the
/// `TxSize` enum ordering used by [`LoopFilterMask`].
const TX_8X8: usize = 1;
/// Index of `TX_16X16` in the per-TX-size mask arrays.
const TX_16X16: usize = 2;
/// Index of `TX_32X32` in the per-TX-size mask arrays.
const TX_32X32: usize = 3;
/// Index of `TX_4X4` in the per-TX-size mask arrays.
const TX_4X4: usize = 0;

/// Final adjustment pass on a 64×64 [`LoopFilterMask`]: merge
/// TX_32X32 into TX_16X16, promote 4×4-border edges to 8×8, mask
/// off frame-edge cells, and clear the left-frame-edge bit on
/// column 0.
///
/// `mi_row` / `mi_col` are the top-left 8×8 coordinates of the
/// superblock in the frame's MI grid; `mi_rows` / `mi_cols` are
/// the full-frame MI extents. `lfm` is the per-superblock mask
/// produced by [`super::setup_mask::setup_mask`] (the
/// partition-tree walker); it is modified in place.
///
/// Mirrors libvpx `vp9_adjust_mask` (`vp9_loopfilter.c:766-878`).
pub fn adjust_mask(mi_row: u32, mi_col: u32, mi_rows: u32, mi_cols: u32, lfm: &mut LoopFilterMask) {
    // 1. Merge TX_32X32 contributions into TX_16X16 (no 32-tap
    //    filter exists; the largest kernel is 16-tap).
    //    `vp9_loopfilter.c:770-775`.
    lfm.left_y[TX_16X16] |= lfm.left_y[TX_32X32];
    lfm.above_y[TX_16X16] |= lfm.above_y[TX_32X32];
    lfm.left_uv[TX_16X16] |= lfm.left_uv[TX_32X32];
    lfm.above_uv[TX_16X16] |= lfm.above_uv[TX_32X32];

    // 2. Promote 4×4 border edges to 8×8. Any TX_4X4 bit on a
    //    per-8×8 left/above border becomes a TX_8X8 bit and is
    //    cleared from TX_4X4. `vp9_loopfilter.c:777-787`.
    lfm.left_y[TX_8X8] |= lfm.left_y[TX_4X4] & LEFT_BORDER_Y;
    lfm.left_y[TX_4X4] &= !LEFT_BORDER_Y;
    lfm.above_y[TX_8X8] |= lfm.above_y[TX_4X4] & ABOVE_BORDER_Y;
    lfm.above_y[TX_4X4] &= !ABOVE_BORDER_Y;
    lfm.left_uv[TX_8X8] |= lfm.left_uv[TX_4X4] & LEFT_BORDER_UV;
    lfm.left_uv[TX_4X4] &= !LEFT_BORDER_UV;
    lfm.above_uv[TX_8X8] |= lfm.above_uv[TX_4X4] & ABOVE_BORDER_UV;
    lfm.above_uv[TX_4X4] &= !ABOVE_BORDER_UV;

    // 3a. Bottom-frame-edge handling: rows past `mi_rows` get
    //     masked off. When the visible row count is exactly 1 or
    //     5 8×8s, the wide (16-tap) UV filter is downgraded to
    //     8-tap because the wider filter would otherwise read
    //     past the frame. `vp9_loopfilter.c:789-817`.
    if mi_row + MI_BLOCK_SIZE > mi_rows {
        let rows = u64::from(mi_rows - mi_row);

        // Every visible 8×8 row contributes 8 bits to the Y mask
        // (one column per 8×8). `mask_y = (1 << (rows*8)) - 1`.
        let mask_y: u64 = (1u64 << (rows << 3)) - 1;
        // Chroma: every two luma 8×8 rows = one 4×4 UV row, so
        // the UV row count is `(rows + 1) / 2`, contributing
        // 4 bits per row. Compute in u32 so the `rows == 7` case
        // (shift amount = 16) doesn't overflow `u16`; the result
        // truncates cleanly back to `u16` because the top bit of
        // the u32 result is just past the u16 boundary
        // (1<<16 - 1 = 0xffff fits in u16).
        let mask_uv: u16 =
            ((1u32 << (((rows + 1) >> 1) << 2)) - 1) as u16;

        // Mask off out-of-frame bits for TX_4X4 / TX_8X8 /
        // TX_16X16. The TX_32X32 slot is *not* masked — the apply
        // pass reads TX_16X16 (which already absorbed TX_32X32),
        // and libvpx's loop here matches `i < TX_32X32`
        // (`:798, 832`).
        for i in 0..TX_32X32 {
            lfm.left_y[i] &= mask_y;
            lfm.above_y[i] &= mask_y;
            lfm.left_uv[i] &= mask_uv;
            lfm.above_uv[i] &= mask_uv;
        }
        lfm.int_4x4_y &= mask_y;
        lfm.int_4x4_uv &= mask_uv;

        // Downgrade the wide UV filter on the very last row.
        if rows == 1 {
            // Only one luma 8×8 row visible → one UV 4×4 row.
            // Any TX_16X16 above bit in that row becomes TX_8X8.
            lfm.above_uv[TX_8X8] |= lfm.above_uv[TX_16X16];
            lfm.above_uv[TX_16X16] = 0;
        }
        if rows == 5 {
            // Three UV 4×4 rows visible (rows 0/1/2 of UV mask).
            // The last UV row (bits 0x_ff00 = UV row 2) gets its
            // TX_16X16 bits demoted to TX_8X8. Libvpx writes the
            // clear as `&= ~(x & 0xff00)`; identically `&= ~0xff00`
            // since `~(A & 0xff00) & x` collapses to `x & ~0xff00`.
            lfm.above_uv[TX_8X8] |= lfm.above_uv[TX_16X16] & 0xff00;
            lfm.above_uv[TX_16X16] &= !0xff00;
        }
    }

    // 3b. Right-frame-edge handling: columns past `mi_cols` get
    //     masked off. Same wide-UV-downgrade for `columns ∈ {1, 5}`.
    //     `vp9_loopfilter.c:819-851`.
    if mi_col + MI_BLOCK_SIZE > mi_cols {
        let columns = u64::from(mi_cols - mi_col);

        // For Y: a 1-bit-per-column pattern broadcast across all 8
        // luma rows. `mask_y = ((1 << columns) - 1) * 0x01010101_01010101`.
        let mask_y: u64 = ((1u64 << columns) - 1).wrapping_mul(0x0101_0101_0101_0101);
        // For UV: same idea on a 4-row × 4-col grid. UV columns =
        // (columns + 1) / 2.
        let mask_uv: u16 =
            ((1u16 << ((columns + 1) >> 1)) - 1).wrapping_mul(0x1111);
        // Internal-4×4 chroma edges do *not* apply on the last
        // visible column — that border is the implicit
        // frame-edge, which is never filtered. Hence a strictly
        // smaller mask.
        let mask_uv_int: u16 =
            ((1u16 << (columns >> 1)) - 1).wrapping_mul(0x1111);

        for i in 0..TX_32X32 {
            lfm.left_y[i] &= mask_y;
            lfm.above_y[i] &= mask_y;
            lfm.left_uv[i] &= mask_uv;
            lfm.above_uv[i] &= mask_uv;
        }
        lfm.int_4x4_y &= mask_y;
        lfm.int_4x4_uv &= mask_uv_int;

        // Downgrade wide UV filter on the last column.
        if columns == 1 {
            lfm.left_uv[TX_8X8] |= lfm.left_uv[TX_16X16];
            lfm.left_uv[TX_16X16] = 0;
        }
        if columns == 5 {
            // UV columns 0/1/2 visible. Bits 0xcccc (= the
            // top-half of each nibble: cols 2 and 3 of every UV
            // row — wait, 0xcccc = 0b1100_1100_1100_1100 = col
            // 2/3 of each row). Actually: 0xcccc covers UV col 2
            // and 3 of every row, but col 3 was already masked
            // off by `mask_uv = 0x7777` above, so the only
            // surviving bits there are UV col 2. Demote them.
            lfm.left_uv[TX_8X8] |= lfm.left_uv[TX_16X16] & 0xcccc;
            lfm.left_uv[TX_16X16] &= !0xcccc;
        }
    }

    // 3c. Left-frame-edge clear: column 0 (luma) / column 0 (UV)
    //     edges are never filtered (no pixels to the left of the
    //     frame). `vp9_loopfilter.c:852-859`.
    if mi_col == 0 {
        for i in 0..TX_32X32 {
            lfm.left_y[i] &= 0xfefe_fefe_fefe_fefe;
            lfm.left_uv[i] &= 0xeeee;
        }
    }

    // Assert mutual exclusion across TX-size slots, mirroring the
    // libvpx asserts at `:862-877`. Cheap in release.
    debug_assert_eq!(lfm.left_y[TX_16X16] & lfm.left_y[TX_8X8], 0);
    debug_assert_eq!(lfm.left_y[TX_16X16] & lfm.left_y[TX_4X4], 0);
    debug_assert_eq!(lfm.left_y[TX_8X8] & lfm.left_y[TX_4X4], 0);
    debug_assert_eq!(lfm.int_4x4_y & lfm.left_y[TX_16X16], 0);
    debug_assert_eq!(lfm.left_uv[TX_16X16] & lfm.left_uv[TX_8X8], 0);
    debug_assert_eq!(lfm.left_uv[TX_16X16] & lfm.left_uv[TX_4X4], 0);
    debug_assert_eq!(lfm.left_uv[TX_8X8] & lfm.left_uv[TX_4X4], 0);
    debug_assert_eq!(lfm.int_4x4_uv & lfm.left_uv[TX_16X16], 0);
    debug_assert_eq!(lfm.above_y[TX_16X16] & lfm.above_y[TX_8X8], 0);
    debug_assert_eq!(lfm.above_y[TX_16X16] & lfm.above_y[TX_4X4], 0);
    debug_assert_eq!(lfm.above_y[TX_8X8] & lfm.above_y[TX_4X4], 0);
    debug_assert_eq!(lfm.int_4x4_y & lfm.above_y[TX_16X16], 0);
    debug_assert_eq!(lfm.above_uv[TX_16X16] & lfm.above_uv[TX_8X8], 0);
    debug_assert_eq!(lfm.above_uv[TX_16X16] & lfm.above_uv[TX_4X4], 0);
    debug_assert_eq!(lfm.above_uv[TX_8X8] & lfm.above_uv[TX_4X4], 0);
    debug_assert_eq!(lfm.int_4x4_uv & lfm.above_uv[TX_16X16], 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::TX_SIZES;

    /// Build a [`LoopFilterMask`] from explicit per-TX-size arrays.
    /// Convenience for setting up hand-traced inputs.
    fn mk_mask(
        left_y: [u64; TX_SIZES],
        above_y: [u64; TX_SIZES],
        int_4x4_y: u64,
        left_uv: [u16; TX_SIZES],
        above_uv: [u16; TX_SIZES],
        int_4x4_uv: u16,
    ) -> LoopFilterMask {
        LoopFilterMask {
            left_y,
            above_y,
            int_4x4_y,
            left_uv,
            above_uv,
            int_4x4_uv,
            lfl_y: [0; 64],
        }
    }

    /// Interior SB — every adjustment branch is either a no-op
    /// for the inputs we set, or the SB isn't on any frame edge.
    /// The mask passes through unchanged.
    ///
    /// Inputs:
    /// - `mi_row = 8`, `mi_col = 8` (middle of a 24×24-MI frame).
    /// - No `TX_32X32` bits set (merge step OR's in 0 → no-op).
    /// - No `TX_4X4` bits on left/above borders (promote step
    ///   AND's against 0 → no-op).
    ///
    /// Expected: input == output.
    #[test]
    fn adjust_mask_interior_sb_passthrough() {
        // TX_8X8 left-edges at row 0 column 1 (bit pos 1) and
        // TX_16X16 above-edges at row 0 column 4 (bit pos 4).
        // Neither overlaps with left_border (0x1111…) for the
        // TX_4X4 slot, which we keep empty.
        let mut lfm = mk_mask(
            [0, 0x0000_0000_0000_0002, 0, 0],
            [0, 0, 0x0000_0000_0000_0010, 0],
            0,
            [0, 0x0002, 0, 0],
            [0, 0, 0x0010, 0],
            0,
        );
        let before = lfm;
        adjust_mask(8, 8, 24, 24, &mut lfm);
        assert_eq!(lfm, before, "interior SB must pass through unchanged");
    }

    /// Right-edge SB: `mi_col + 8 > mi_cols` triggers the
    /// columns-mask path. With a TX_32X32 region present, the
    /// merge step demotes it to TX_16X16, then the columns-mask
    /// clears bits past the visible column count.
    ///
    /// Setup: 12-col frame, SB at `mi_col=8` → `columns = 4`
    /// visible. `mask_y = ((1<<4)-1) * 0x0101...01 = 0x0f0f_..._0f0f`.
    /// Place a 64×64 TX_32X32 contribution
    /// (`above_y[TX_32X32] = 0x000000ff_000000ff`,
    /// `left_y[TX_32X32] = 0x1111_..._1111`) plus a `lfl_y`
    /// witness; verify merge + right-clamp.
    #[test]
    fn adjust_mask_right_edge_tx32_demoted_and_clamped() {
        let mut lfm = LoopFilterMask {
            left_y: [0, 0, 0, 0x1111_1111_1111_1111],
            above_y: [0, 0, 0, 0x0000_00ff_0000_00ff],
            int_4x4_y: 0,
            left_uv: [0, 0, 0, 0x1111],
            above_uv: [0, 0, 0, 0x000f],
            int_4x4_uv: 0,
            lfl_y: [42; 64],
        };
        adjust_mask(0, 8, 8, 12, &mut lfm);

        // Step 1 (merge) moves TX_32X32 into TX_16X16. TX_32X32
        // itself is left as-is (libvpx leaves it; apply pass
        // reads TX_16X16).
        //
        // Step 3b (right-edge) then ANDs every TX_<32 slot with
        // mask_y = 0x0f0f_0f0f_0f0f_0f0f (Y) / mask_uv = 0x0033
        // (UV, since (4+1)>>1 = 2 → (1<<8)-1 = 0xff, then *0x1111
        // -> only the first two UV columns survive).
        //
        // Hand-traced:
        //   columns = 4
        //   mask_y  = ((1 << 4) - 1) * 0x0101_..._0101
        //           = 0xf * 0x0101_..._0101
        //           = 0x0f0f_0f0f_0f0f_0f0f
        //   mask_uv = ((1 << 2) - 1) * 0x1111 = 3 * 0x1111 = 0x3333
        //   mask_uv_int = ((1 << 2) - 1) * 0x1111 = 0x3333
        //
        // left_y[TX_16X16] after merge:  0x1111_..._1111
        // left_y[TX_16X16] after &mask_y: 0x0101_0101_0101_0101
        assert_eq!(lfm.left_y[TX_16X16], 0x0101_0101_0101_0101);
        // above_y[TX_16X16] after merge: 0x0000_00ff_0000_00ff
        // after &mask_y:                 0x0000_000f_0000_000f
        assert_eq!(lfm.above_y[TX_16X16], 0x0000_000f_0000_000f);
        // TX_32X32 slot left untouched by step 3 (i < TX_32X32 in libvpx).
        assert_eq!(lfm.left_y[TX_32X32], 0x1111_1111_1111_1111);
        assert_eq!(lfm.above_y[TX_32X32], 0x0000_00ff_0000_00ff);
        // UV: left_uv[TX_16X16] = 0x1111 → after &mask_uv (0x3333) = 0x1111.
        //     above_uv[TX_16X16] = 0x000f → after &mask_uv (0x3333) = 0x0003.
        assert_eq!(lfm.left_uv[TX_16X16], 0x1111);
        assert_eq!(lfm.above_uv[TX_16X16], 0x0003);
        // lfl_y untouched (adjust_mask doesn't write filter-level array).
        assert_eq!(lfm.lfl_y, [42; 64]);
    }

    /// Bottom-edge SB: `mi_row + 8 > mi_rows` triggers the
    /// rows-mask path. With `rows = 2`, only the top two 8×8
    /// luma rows survive (`mask_y = 0x0000_0000_0000_ffff`) and
    /// only the top UV row survives (`mask_uv = 0x000f`).
    ///
    /// Place the SB at `mi_col = 8` so the left-edge clear (only
    /// fires on `mi_col == 0`) does not interfere with the
    /// bottom-edge clamp under test.
    ///
    /// Bit positions are chosen to satisfy the libvpx mutual-
    /// exclusion invariant (no overlap between TX-size slots,
    /// no overlap between `int_4x4_*` and `[TX_16X16]`).
    #[test]
    fn adjust_mask_bottom_edge_clamps_rows() {
        let mut lfm = LoopFilterMask {
            // left_y at TX_8X8 spans all 8 rows; above_y at
            // TX_16X16 spans all 8 rows but only every second
            // bit (so it can't overlap int_4x4_y bits). int_4x4_y
            // gets a disjoint pattern. The simplest disjoint
            // assignment: above_y[TX_16X16] uses 0xff bit pattern
            // per row position 0; int_4x4_y uses bit position 1
            // of every row.
            left_y: [0, 0xffff_ffff_ffff_ffff, 0, 0],
            above_y: [0, 0, 0x0101_0101_0101_0101, 0],
            int_4x4_y: 0x0202_0202_0202_0202,
            // Same idea for UV. above_uv[TX_16X16] at bit 0 of each
            // row (0x1111); int_4x4_uv at bit 1 of each row
            // (0x2222) — disjoint.
            left_uv: [0, 0xffff, 0, 0],
            above_uv: [0, 0, 0x1111, 0],
            int_4x4_uv: 0x2222,
            lfl_y: [0; 64],
        };
        adjust_mask(8, 8, 10, 24, &mut lfm);
        // rows = 2:
        //   mask_y = (1 << 16) - 1 = 0x0000_0000_0000_ffff.
        //   mask_uv = (1 << (((2+1) >> 1) << 2)) - 1 = (1 << 4) - 1 = 0x000f.
        // After AND:
        //   left_y[TX_8X8]   = 0xffff_ffff_ffff_ffff & 0xffff = 0xffff
        //   above_y[TX_16X16]= 0x0101_..._0101 & 0xffff = 0x0101
        //   int_4x4_y        = 0x0202_..._0202 & 0xffff = 0x0202
        //   left_uv[TX_8X8]  = 0xffff & 0x000f = 0x000f
        //   above_uv[TX_16X16]= 0x1111 & 0x000f = 0x0001
        //   int_4x4_uv       = 0x2222 & 0x000f = 0x0002
        assert_eq!(lfm.left_y[TX_8X8], 0x0000_0000_0000_ffff);
        assert_eq!(lfm.above_y[TX_16X16], 0x0000_0000_0000_0101);
        assert_eq!(lfm.int_4x4_y, 0x0000_0000_0000_0202);
        assert_eq!(lfm.left_uv[TX_8X8], 0x000f);
        assert_eq!(lfm.above_uv[TX_16X16], 0x0001);
        assert_eq!(lfm.int_4x4_uv, 0x0002);
        // rows == 2 → no `rows == 1` or `rows == 5` branch fires.
    }

    /// Left-frame-edge SB: `mi_col == 0` triggers the
    /// left-column clear. Every `left_y[i]` slot gets its
    /// column-0 bits zeroed (mask `0xfefe…`) and every
    /// `left_uv[i]` slot gets its column-0 bits zeroed (mask
    /// `0xeeee`). `above_*` and `int_4x4_*` are untouched by
    /// this step.
    #[test]
    fn adjust_mask_left_frame_edge_clears_col0() {
        // Populate left edges with all-ones; populate above/int
        // to verify they survive. No TX_32X32 / TX_4X4 border
        // bits, so steps 1 and 2 are no-ops; mi_row=0/mi_col=0
        // with mi_rows=mi_cols=8 → no right/bottom edge triggers.
        let mut lfm = LoopFilterMask {
            left_y: [
                0,
                0xffff_ffff_ffff_ffff,
                0,
                0,
            ],
            above_y: [0, 0xffff_ffff_ffff_ffff, 0, 0],
            int_4x4_y: 0xffff_ffff_ffff_ffff,
            left_uv: [0, 0xffff, 0, 0],
            above_uv: [0, 0xffff, 0, 0],
            int_4x4_uv: 0xffff,
            lfl_y: [0; 64],
        };
        adjust_mask(0, 0, 8, 8, &mut lfm);
        // Column-0 bits cleared in every left_y[i]. Mask
        // 0xfefe… clears bit 0 of every byte → bit 0 of every
        // 8×8 row.
        assert_eq!(lfm.left_y[TX_8X8], 0xfefe_fefe_fefe_fefe);
        // Column-0 bits cleared in every left_uv[i]. Mask 0xeeee
        // clears bit 0 of every nibble → bit 0 of every UV row.
        assert_eq!(lfm.left_uv[TX_8X8], 0xeeee);
        // above / int untouched.
        assert_eq!(lfm.above_y[TX_8X8], 0xffff_ffff_ffff_ffff);
        assert_eq!(lfm.above_uv[TX_8X8], 0xffff);
        assert_eq!(lfm.int_4x4_y, 0xffff_ffff_ffff_ffff);
        assert_eq!(lfm.int_4x4_uv, 0xffff);
    }

    /// 4×4-border promotion: `TX_4X4` bits on the left-edge
    /// border (every 8th bit) and above-edge border (every byte's
    /// LSBs) get promoted to `TX_8X8` and cleared from `TX_4X4`.
    ///
    /// Setup: left_y[TX_4X4] = LEFT_BORDER_Y | (one off-border
    /// bit). Verify TX_8X8 picks up the border bits and TX_4X4
    /// keeps the off-border bit.
    #[test]
    fn adjust_mask_promotes_4x4_borders_to_8x8() {
        let off_border_bit: u64 = 1 << 2; // row 0, col 2 — not on left/above border.
        let mut lfm = mk_mask(
            // left_y: border bits + one off-border bit at TX_4X4.
            [LEFT_BORDER_Y | off_border_bit, 0, 0, 0],
            // above_y: border bits at TX_4X4.
            [ABOVE_BORDER_Y, 0, 0, 0],
            0,
            [LEFT_BORDER_UV, 0, 0, 0],
            [ABOVE_BORDER_UV, 0, 0, 0],
            0,
        );
        // Interior SB (no edge triggers).
        adjust_mask(8, 8, 24, 24, &mut lfm);

        // TX_8X8 picks up exactly the border bits.
        assert_eq!(lfm.left_y[TX_8X8], LEFT_BORDER_Y);
        assert_eq!(lfm.above_y[TX_8X8], ABOVE_BORDER_Y);
        assert_eq!(lfm.left_uv[TX_8X8], LEFT_BORDER_UV);
        assert_eq!(lfm.above_uv[TX_8X8], ABOVE_BORDER_UV);

        // TX_4X4 keeps only the off-border bit.
        assert_eq!(lfm.left_y[TX_4X4], off_border_bit);
        assert_eq!(lfm.above_y[TX_4X4], 0);
        assert_eq!(lfm.left_uv[TX_4X4], 0);
        assert_eq!(lfm.above_uv[TX_4X4], 0);
    }

    /// Right + bottom corner SB with `columns == 1` and `rows == 1`.
    /// Both edge branches fire: the 16×16 UV filter is downgraded
    /// to 8×8 on both the last column and the last row.
    ///
    /// Setup: 9-col × 9-row frame. SB at (8, 8) → `rows = 1`,
    /// `columns = 1`. With TX_16X16 UV bits set, the columns==1
    /// branch demotes left_uv[TX_16X16] → left_uv[TX_8X8] and
    /// zeroes left_uv[TX_16X16]. Similarly, rows==1 demotes
    /// above_uv[TX_16X16] → above_uv[TX_8X8].
    #[test]
    fn adjust_mask_corner_columns1_rows1_uv_downgrade() {
        let mut lfm = LoopFilterMask {
            left_y: [0; TX_SIZES],
            above_y: [0; TX_SIZES],
            int_4x4_y: 0,
            // left_uv at TX_16X16: column 0 of row 0 (= bit 0,
            // which survives the columns=1 mask_uv=0x1111).
            left_uv: [0, 0, 0x0001, 0],
            // above_uv at TX_16X16: column 0 of row 0 (= bit 0,
            // which survives the rows=1 mask_uv=0x000f).
            above_uv: [0, 0, 0x0001, 0],
            int_4x4_uv: 0,
            lfl_y: [0; 64],
        };
        adjust_mask(8, 8, 9, 9, &mut lfm);

        // Row-branch (rows == 1): above_uv[TX_16X16] cleared;
        // its bit moved to above_uv[TX_8X8].
        assert_eq!(lfm.above_uv[TX_16X16], 0);
        assert_eq!(lfm.above_uv[TX_8X8], 0x0001);

        // Column-branch (columns == 1): left_uv[TX_16X16]
        // cleared; its bit moved to left_uv[TX_8X8].
        assert_eq!(lfm.left_uv[TX_16X16], 0);
        assert_eq!(lfm.left_uv[TX_8X8], 0x0001);
    }

    /// Right-edge SB with `columns == 5`: the wide UV filter on
    /// the last visible UV column (= UV column 2, bits 0xcccc
    /// limited by mask_uv = 0x7777 → effectively bits 0x4444)
    /// gets demoted from TX_16X16 to TX_8X8.
    ///
    /// Setup: 13-col frame, SB at mi_col = 8 → `columns = 5`.
    /// `mask_uv = ((1 << 3) - 1) * 0x1111 = 0x7777`. Set
    /// `left_uv[TX_16X16] = 0x4444` (= UV col 2 of every row);
    /// after mask, it survives. The `columns == 5` branch then
    /// demotes its overlap with `0xcccc` (= 0x4444) to TX_8X8.
    #[test]
    fn adjust_mask_right_edge_columns5_uv_col2_downgrade() {
        let mut lfm = LoopFilterMask {
            left_y: [0; TX_SIZES],
            above_y: [0; TX_SIZES],
            int_4x4_y: 0,
            left_uv: [0, 0, 0x4444, 0],
            above_uv: [0; TX_SIZES],
            int_4x4_uv: 0,
            lfl_y: [0; 64],
        };
        adjust_mask(0, 8, 8, 13, &mut lfm);

        // After &mask_uv (0x7777): left_uv[TX_16X16] = 0x4444.
        // After `if columns == 5`:
        //   left_uv[TX_8X8] |= 0x4444 & 0xcccc = 0x4444
        //   left_uv[TX_16X16] &= !0xcccc = 0x4444 & 0x3333 = 0
        assert_eq!(lfm.left_uv[TX_8X8], 0x4444);
        assert_eq!(lfm.left_uv[TX_16X16], 0);
    }

    /// Bottom-edge SB with `rows == 5`: the wide UV filter on
    /// the last visible UV row (= UV row 2, bits 0x_ff00 limited
    /// by mask_uv = 0x0fff → effectively bits 0x0f00) gets
    /// demoted from TX_16X16 to TX_8X8.
    #[test]
    fn adjust_mask_bottom_edge_rows5_uv_row2_downgrade() {
        let mut lfm = LoopFilterMask {
            left_y: [0; TX_SIZES],
            above_y: [0; TX_SIZES],
            int_4x4_y: 0,
            left_uv: [0; TX_SIZES],
            // above_uv at TX_16X16: UV row 2 (bits 0x0f00).
            above_uv: [0, 0, 0x0f00, 0],
            int_4x4_uv: 0,
            // Place SB at mi_col=8 so the left-edge clear doesn't fire.
            lfl_y: [0; 64],
        };
        adjust_mask(8, 8, 13, 24, &mut lfm);

        // rows = 5, mask_uv = ((1 << ((5+1)/2)) - 1) * ... wait:
        //   ((rows + 1) >> 1) << 2 = (6/2) << 2 = 3 << 2 = 12
        //   mask_uv = (1 << 12) - 1 = 0x0fff
        // So above_uv[TX_16X16] = 0x0f00 & 0x0fff = 0x0f00.
        // Then `if rows == 5`:
        //   above_uv[TX_8X8] |= 0x0f00 & 0xff00 = 0x0f00
        //   above_uv[TX_16X16] &= !0xff00 = 0x0f00 & 0x00ff = 0
        assert_eq!(lfm.above_uv[TX_8X8], 0x0f00);
        assert_eq!(lfm.above_uv[TX_16X16], 0);
    }

    /// Empty mask in, empty mask out — every step is a no-op.
    #[test]
    fn adjust_mask_zero_in_zero_out() {
        let mut lfm = LoopFilterMask::ZERO;
        adjust_mask(0, 0, 8, 8, &mut lfm);
        // mi_col == 0 fires the left-edge clear, but ANDing zero
        // with anything stays zero.
        assert_eq!(lfm, LoopFilterMask::ZERO);
    }
}
