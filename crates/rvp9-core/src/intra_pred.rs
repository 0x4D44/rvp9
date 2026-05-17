//! VP9 intra prediction kernels.
//!
//! Mirrors libvpx `vpx_dsp/intrapred.c` for the 8-bit (non-HBD) path.
//! Each predictor reads from `above` and/or `left` neighbour rows and
//! writes a `bs × bs` block into `dst` with the given `stride`.
//!
//! ## Scope of this commit (Task 5.1)
//!
//! Lands the **four primary intra modes** plus the **three DC
//! fallback variants** for the 8-bit path:
//!
//! - [`v_predictor`] — V_PRED, vertical (copy `above` row).
//! - [`h_predictor`] — H_PRED, horizontal (copy `left[r]` across row).
//! - [`tm_predictor`] — TM_PRED (true motion: `left[r] + above[c] -
//!   above_left`, clipped).
//! - [`dc_predictor`] — DC_PRED (average of `above` and `left`).
//! - [`dc_128_predictor`] — DC fallback when no neighbours are
//!   available (border / first block).
//! - [`dc_left_predictor`] — DC fallback when only `left` is available.
//! - [`dc_top_predictor`] — DC fallback when only `above` is available.
//!
//! All work at any `bs ∈ {4, 8, 16, 32}` (libvpx passes `bs` runtime).
//!
//! ## Directional modes
//!
//! Task 5.2 adds three of the six directional intra modes —
//! [`d45_predictor`], [`d63_predictor`], [`d207_predictor`] — that
//! use only `above` OR only `left` neighbours. The three corner
//! directionals (`D117 / D135 / D153`) need the top-left pixel
//! plus `above` AND `left`; they land in Task 5.3.
//!
//! Directional kernels read `above` pixels past the current block
//! (i.e. `above[bs..2*bs]` — the "above-right" pixels of the block
//! to the right of the current one). Callers must populate a
//! `2 * bs`-byte `above` slice when invoking a directional predictor;
//! if the right neighbour hasn't been decoded yet, callers pad with
//! the rightmost real `above` pixel (libvpx
//! `vp9_intra_predict_top_pixels` convention).
//!
//! ## API conventions
//!
//! libvpx's C kernels read `above[-1]` to grab the top-left corner.
//! That requires an unsafe pointer dance in Rust. We make the
//! top-left an explicit parameter [`PredCtx::above_left`] so all
//! borrows are bounded slices. The helper [`PredCtx::from_libvpx_style`]
//! constructs one from libvpx's `above` slice (assuming the
//! convention that `above` starts at the top-left pixel — equivalent
//! to libvpx's `above[-1]`).

/// Inputs to a per-block intra predictor. The block above and the
/// block to the left are already-decoded neighbours; `above_left` is
/// the top-left corner pixel shared by both.
#[derive(Copy, Clone, Debug)]
pub struct PredCtx<'a> {
    /// `bs` pixels of the row directly above the block.
    pub above: &'a [u8],
    /// `bs` pixels of the column directly to the left of the block.
    pub left: &'a [u8],
    /// The pixel at the top-left corner (one diagonal step away from
    /// the block's top-left).
    pub above_left: u8,
}

impl<'a> PredCtx<'a> {
    /// Convenience: build a context from a libvpx-style "above
    /// pointer" + "left pointer" where the caller has access to
    /// `above[-1]` (the byte preceding the `bs` `above` pixels —
    /// matches libvpx's pointer convention).
    #[inline]
    #[must_use]
    pub fn from_libvpx_style(above_with_corner: &'a [u8], left: &'a [u8]) -> Self {
        debug_assert!(
            !above_with_corner.is_empty(),
            "above slice must include the top-left corner at index 0"
        );
        Self {
            above_left: above_with_corner[0],
            above: &above_with_corner[1..],
            left,
        }
    }
}

/// V_PRED — copy the `above` row down across all `bs` rows.
/// Mirrors libvpx `v_predictor` (`intrapred.c:157-166`).
pub fn v_predictor(dst: &mut [u8], stride: usize, bs: usize, ctx: PredCtx<'_>) {
    debug_assert!(ctx.above.len() >= bs);
    for r in 0..bs {
        dst[r * stride..r * stride + bs].copy_from_slice(&ctx.above[..bs]);
    }
}

/// H_PRED — broadcast `left[r]` across row `r`.
/// Mirrors libvpx `h_predictor` (`intrapred.c:168-177`).
pub fn h_predictor(dst: &mut [u8], stride: usize, bs: usize, ctx: PredCtx<'_>) {
    debug_assert!(ctx.left.len() >= bs);
    for r in 0..bs {
        let v = ctx.left[r];
        dst[r * stride..r * stride + bs].fill(v);
    }
}

/// TM_PRED — "true motion": `clip(left[r] + above[c] - above_left)`.
/// Mirrors libvpx `tm_predictor` (`intrapred.c:179-189`).
pub fn tm_predictor(dst: &mut [u8], stride: usize, bs: usize, ctx: PredCtx<'_>) {
    debug_assert!(ctx.above.len() >= bs);
    debug_assert!(ctx.left.len() >= bs);
    let tl = ctx.above_left as i32;
    for r in 0..bs {
        let l = ctx.left[r] as i32;
        let row = &mut dst[r * stride..r * stride + bs];
        for (c, slot) in row.iter_mut().enumerate() {
            let v = l + ctx.above[c] as i32 - tl;
            *slot = v.clamp(0, 255) as u8;
        }
    }
}

/// DC_PRED — block-wide constant equal to the rounded average of
/// the `2 * bs` neighbour pixels. Mirrors libvpx `dc_predictor`
/// (`intrapred.c:232-248`).
pub fn dc_predictor(dst: &mut [u8], stride: usize, bs: usize, ctx: PredCtx<'_>) {
    debug_assert!(ctx.above.len() >= bs);
    debug_assert!(ctx.left.len() >= bs);
    let count = 2 * bs;
    let mut sum: u32 = 0;
    for i in 0..bs {
        sum += ctx.above[i] as u32;
        sum += ctx.left[i] as u32;
    }
    let half = (count as u32) >> 1;
    let dc = ((sum + half) / count as u32) as u8;
    fill_block(dst, stride, bs, dc);
}

/// DC fallback used when neither `above` nor `left` is available
/// (border, picture origin). The constant is 128 — the
/// maximum-entropy mid-grey. Mirrors libvpx `dc_128_predictor`
/// (`intrapred.c:191-201`).
pub fn dc_128_predictor(dst: &mut [u8], stride: usize, bs: usize) {
    fill_block(dst, stride, bs, 128);
}

/// DC fallback used when only `left` is available. Mirrors libvpx
/// `dc_left_predictor` (`intrapred.c:203-216`).
pub fn dc_left_predictor(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    debug_assert!(left.len() >= bs);
    let sum: u32 = left.iter().take(bs).map(|&x| x as u32).sum();
    let half = (bs as u32) >> 1;
    let dc = ((sum + half) / bs as u32) as u8;
    fill_block(dst, stride, bs, dc);
}

/// DC fallback used when only `above` is available. Mirrors libvpx
/// `dc_top_predictor` (`intrapred.c:218-230`).
pub fn dc_top_predictor(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    debug_assert!(above.len() >= bs);
    let sum: u32 = above.iter().take(bs).map(|&x| x as u32).sum();
    let half = (bs as u32) >> 1;
    let dc = ((sum + half) / bs as u32) as u8;
    fill_block(dst, stride, bs, dc);
}

/// Helper: write `value` across a `bs × bs` block at `dst` with
/// `stride`.
#[inline]
fn fill_block(dst: &mut [u8], stride: usize, bs: usize, value: u8) {
    for r in 0..bs {
        dst[r * stride..r * stride + bs].fill(value);
    }
}

// --------------------------------------------------------------- directional
//
// Helpers mirroring libvpx's macros in `vpx_dsp/intrapred.c:17-19`:
//   AVG3(a, b, c) = ((a) + 2 * (b) + (c) + 2) >> 2
//   AVG2(a, b)    = ((a) + (b) + 1) >> 1
//
// We use `u32` accumulators to avoid any overflow during the sum and
// then cast back to `u8` — values are always in `[0, 255]` by
// construction.

#[inline]
const fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u32 + 2 * b as u32 + c as u32 + 2) >> 2) as u8
}

#[inline]
const fn avg2(a: u8, b: u8) -> u8 {
    ((a as u32 + b as u32 + 1) >> 1) as u8
}

/// D45 — predict using a 45-degree downward-right edge from the
/// top-row neighbours. Reads `above[0..=2*bs - 1]` (so `above` must
/// have at least `2 * bs` pixels). Does not use `left` or `above_left`.
///
/// Mirrors libvpx `d45_predictor` (`intrapred.c:62-79`).
pub fn d45_predictor(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    debug_assert!(above.len() >= 2 * bs);
    let above_right = above[bs - 1];
    // Row 0: AVG3(above[x], above[x+1], above[x+2]) for x in 0..bs-1;
    // last pixel is just above_right.
    for x in 0..bs - 1 {
        dst[x] = avg3(above[x], above[x + 1], above[x + 2]);
    }
    dst[bs - 1] = above_right;
    // Each subsequent row shifts left by 1, padding with above_right.
    // Row r is row 0 shifted left by r, with the trailing r+1 pixels
    // replaced by above_right.
    for r in 1..bs {
        let size = bs - r - 1;
        let row_start = r * stride;
        let (head, tail) = dst.split_at_mut(row_start);
        if size > 0 {
            tail[..size].copy_from_slice(&head[r..r + size]);
        }
        tail[size..bs].fill(above_right);
    }
}

/// D63 — predict using a 63-degree downward-right edge. Reads
/// `above[0..=bs + 1]` minimum (the `c + 2 == bs + 1` boundary at
/// `c = bs - 1`); caller should supply `2 * bs` pixels for safety.
/// Does not use `left` or `above_left`.
///
/// Mirrors libvpx `d63_predictor` (`intrapred.c:45-60`).
pub fn d63_predictor(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    debug_assert!(above.len() >= 2 * bs);
    // Rows 0 and 1 are filtered from above.
    for c in 0..bs {
        dst[c] = avg2(above[c], above[c + 1]);
        dst[stride + c] = avg3(above[c], above[c + 1], above[c + 2]);
    }
    // Rows 2..bs alternate between copying from row 0 / row 1 with a
    // shift, padding the right with above[bs - 1].
    let pad = above[bs - 1];
    let mut size = bs - 2;
    let mut r: usize = 2;
    while r < bs {
        let copy_src_row_0 = r >> 1;
        let copy_src_row_1 = 1 + (r >> 1);
        // Row r (even iteration): copy from row 0's offset
        // (r/2)..(r/2 + size).
        {
            let (head, tail) = dst.split_at_mut(r * stride);
            if size > 0 {
                tail[..size].copy_from_slice(
                    &head[copy_src_row_0..copy_src_row_0 + size],
                );
            }
            tail[size..bs].fill(pad);
        }
        // Row r+1: copy from row 1's offset (r/2)..(r/2 + size). Row 1
        // begins at byte `stride` of dst, and we want offset
        // `copy_src_row_1 - 1`.
        {
            let (head, tail) = dst.split_at_mut((r + 1) * stride);
            let src_start = stride + copy_src_row_1 - 1;
            if size > 0 {
                tail[..size].copy_from_slice(&head[src_start..src_start + size]);
            }
            tail[size..bs].fill(pad);
        }
        r += 2;
        size -= 1;
    }
}

/// D207 — predict using a 207-degree (i.e. 27-degree above horizontal,
/// downward-left) edge. Uses only `left`. Reads `left[0..bs]`.
///
/// Mirrors libvpx `d207_predictor` (`intrapred.c:21-43`).
pub fn d207_predictor(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    debug_assert!(left.len() >= bs);
    // First column: AVG2 along left; last entry just copies left[bs-1].
    for r in 0..bs - 1 {
        dst[r * stride] = avg2(left[r], left[r + 1]);
    }
    dst[(bs - 1) * stride] = left[bs - 1];

    // Second column: AVG3 along left with the same boundary cases.
    for r in 0..bs - 2 {
        dst[r * stride + 1] = avg3(left[r], left[r + 1], left[r + 2]);
    }
    dst[(bs - 2) * stride + 1] = avg3(left[bs - 2], left[bs - 1], left[bs - 1]);
    dst[(bs - 1) * stride + 1] = left[bs - 1];

    // Bottom row, columns 2..bs: pad with left[bs - 1].
    for c in 2..bs {
        dst[(bs - 1) * stride + c] = left[bs - 1];
    }

    // Rows bs-2 .. 0, columns 2..bs: each cell = the cell two columns
    // left in the row below.
    for r in (0..bs - 1).rev() {
        for c in 2..bs {
            // dst[r * stride + c] = dst[(r + 1) * stride + c - 2]
            let src_idx = (r + 1) * stride + c - 2;
            let dst_idx = r * stride + c;
            dst[dst_idx] = dst[src_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dst(bs: usize, stride: usize) -> Vec<u8> {
        vec![0u8; stride * bs]
    }

    #[test]
    fn v_predictor_copies_above_each_row() {
        let above = [10u8, 20, 30, 40];
        let left = [50u8, 60, 70, 80];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 100,
        };
        let mut dst = make_dst(4, 4);
        v_predictor(&mut dst, 4, 4, ctx);
        for r in 0..4 {
            assert_eq!(&dst[r * 4..r * 4 + 4], &[10, 20, 30, 40]);
        }
    }

    #[test]
    fn h_predictor_broadcasts_left_per_row() {
        let above = [10u8, 20, 30, 40];
        let left = [50u8, 60, 70, 80];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 100,
        };
        let mut dst = make_dst(4, 4);
        h_predictor(&mut dst, 4, 4, ctx);
        assert_eq!(&dst[0..4], &[50; 4]);
        assert_eq!(&dst[4..8], &[60; 4]);
        assert_eq!(&dst[8..12], &[70; 4]);
        assert_eq!(&dst[12..16], &[80; 4]);
    }

    #[test]
    fn tm_predictor_hand_computed() {
        // above = [10, 20, 30, 40]; left = [50, 60, 70, 80]; above_left = 25.
        // dst[r][c] = clip(left[r] + above[c] - 25).
        // Row 0: [50+10-25, 50+20-25, 50+30-25, 50+40-25] = [35, 45, 55, 65]
        // Row 1: 60-25 base = 35 → [45, 55, 65, 75]
        // Row 3: 80-25 base = 55 → [65, 75, 85, 95]
        let above = [10u8, 20, 30, 40];
        let left = [50u8, 60, 70, 80];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 25,
        };
        let mut dst = make_dst(4, 4);
        tm_predictor(&mut dst, 4, 4, ctx);
        assert_eq!(&dst[0..4], &[35, 45, 55, 65]);
        assert_eq!(&dst[4..8], &[45, 55, 65, 75]);
        assert_eq!(&dst[8..12], &[55, 65, 75, 85]);
        assert_eq!(&dst[12..16], &[65, 75, 85, 95]);
    }

    #[test]
    fn tm_predictor_clips_high_and_low() {
        let above = [250u8, 250, 250, 250];
        let left = [250u8, 250, 250, 250];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 0,
        };
        let mut dst = make_dst(4, 4);
        tm_predictor(&mut dst, 4, 4, ctx);
        // left[r] + above[c] - 0 = 500 → clipped to 255.
        assert_eq!(dst, [255u8; 16]);

        // Now test underflow: left small, above small, above_left huge.
        let above = [10u8, 10, 10, 10];
        let left = [10u8, 10, 10, 10];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 200,
        };
        let mut dst = make_dst(4, 4);
        tm_predictor(&mut dst, 4, 4, ctx);
        // 10 + 10 - 200 = -180 → clipped to 0.
        assert_eq!(dst, [0u8; 16]);
    }

    #[test]
    fn dc_predictor_rounded_average() {
        // above = [16, 16, 16, 16] sum = 64; left = [32, 32, 32, 32] sum = 128.
        // total sum = 192; count = 8; (192 + 4) / 8 = 24 (libvpx rounds half-up).
        // Actually (192 + 4) / 8 = 196 / 8 = 24.5 → integer division → 24.
        let above = [16u8; 4];
        let left = [32u8; 4];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 0,
        };
        let mut dst = make_dst(4, 4);
        dc_predictor(&mut dst, 4, 4, ctx);
        assert!(dst.iter().all(|&v| v == 24));
    }

    #[test]
    fn dc_128_predictor_writes_128() {
        let mut dst = make_dst(4, 4);
        dc_128_predictor(&mut dst, 4, 4);
        assert_eq!(dst, [128u8; 16]);
    }

    #[test]
    fn dc_left_predictor_averages_left_only() {
        // sum = 0+8+16+24 = 48; (48 + 2) / 4 = 12.
        let left = [0u8, 8, 16, 24];
        let mut dst = make_dst(4, 4);
        dc_left_predictor(&mut dst, 4, 4, &left);
        assert_eq!(dst, [12u8; 16]);
    }

    #[test]
    fn dc_top_predictor_averages_above_only() {
        let above = [100u8, 100, 100, 100];
        let mut dst = make_dst(4, 4);
        dc_top_predictor(&mut dst, 4, 4, &above);
        assert_eq!(dst, [100u8; 16]);
    }

    #[test]
    fn predctx_from_libvpx_style_extracts_corner() {
        // libvpx layout: above slice index 0 is the top-left corner;
        // bs pixels follow at indices 1..=bs.
        let above_with_corner = [99u8, 10, 20, 30, 40];
        let left = [50u8, 60, 70, 80];
        let ctx = PredCtx::from_libvpx_style(&above_with_corner, &left);
        assert_eq!(ctx.above_left, 99);
        assert_eq!(ctx.above, &[10, 20, 30, 40]);
        assert_eq!(ctx.left, &left);
    }

    // -------- Directional kernels --------

    #[test]
    fn d45_predictor_4x4_hand_traced() {
        // above = [A=10, B=20, C=30, D=40, E=50, F=60, G=70, H=80].
        // Row 0: [AVG3(A,B,C), AVG3(B,C,D), AVG3(C,D,E), H]
        //      = [((10+40+30+2)>>2)=20, ((20+60+40+2)>>2)=30,
        //         ((30+80+50+2)>>2)=40, 80]
        //   AVG3(10,20,30) = (10 + 40 + 30 + 2) >> 2 = 82 >> 2 = 20
        //   AVG3(20,30,40) = (20 + 60 + 40 + 2) >> 2 = 122 >> 2 = 30
        //   AVG3(30,40,50) = (30 + 80 + 50 + 2) >> 2 = 162 >> 2 = 40
        //   above_right = above[3] = 40
        // (Re-tracing: above_right = above[bs-1] = above[3] = 40, not 80.)
        let above = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let mut dst = [0u8; 16];
        d45_predictor(&mut dst, 4, 4, &above);
        // Row 0: AVG3(A,B,C)=20, AVG3(B,C,D)=30, AVG3(C,D,E)=40, above_right=40.
        assert_eq!(&dst[0..4], &[20, 30, 40, 40]);
        // Row 1: shift left by 1 → [30, 40, 40, 40].
        assert_eq!(&dst[4..8], &[30, 40, 40, 40]);
        // Row 2: shift left by 2 → [40, 40, 40, 40].
        assert_eq!(&dst[8..12], &[40, 40, 40, 40]);
        // Row 3: all above_right.
        assert_eq!(&dst[12..16], &[40, 40, 40, 40]);
    }

    #[test]
    fn d63_predictor_4x4_hand_traced() {
        // above = [A=0, B=20, C=40, D=60, E=80, F=100, G=120, H=140]
        // Row 0: AVG2 along above pairs:
        //   AVG2(A,B)=10, AVG2(B,C)=30, AVG2(C,D)=50, AVG2(D,E)=70
        // Row 1: AVG3:
        //   AVG3(A,B,C)=(0+40+40+2)>>2=20
        //   AVG3(B,C,D)=(20+80+60+2)>>2=40
        //   AVG3(C,D,E)=(40+120+80+2)>>2=60
        //   AVG3(D,E,F)=(60+160+100+2)>>2=80
        let above = [0u8, 20, 40, 60, 80, 100, 120, 140];
        let mut dst = [0u8; 16];
        d63_predictor(&mut dst, 4, 4, &above);
        assert_eq!(&dst[0..4], &[10, 30, 50, 70]);
        assert_eq!(&dst[4..8], &[20, 40, 60, 80]);
        // Rows 2 and 3 are shifted copies + padding. Pin shapes only.
        // Pad = above[bs-1] = above[3] = 60.
        // r=2: size=2; copy from row 0 starting at offset 1 → [30, 50], pad rest with 60.
        assert_eq!(&dst[8..12], &[30, 50, 60, 60]);
        // r=3: size=2; copy from row 1 starting at offset 1 → [40, 60], pad rest with 60.
        assert_eq!(&dst[12..16], &[40, 60, 60, 60]);
    }

    #[test]
    fn d207_predictor_4x4_hand_traced() {
        // left = [L0=10, L1=30, L2=50, L3=70]
        // Col 0: [AVG2(L0,L1), AVG2(L1,L2), AVG2(L2,L3), L3]
        //      = [(10+30+1)>>1=20, (30+50+1)>>1=40, (50+70+1)>>1=60, 70]
        // Col 1: [AVG3(L0,L1,L2), AVG3(L1,L2,L3), AVG3(L2,L3,L3), L3]
        //      = [(10+60+50+2)>>2=30, (30+100+70+2)>>2=50,
        //         (50+140+70+2)>>2=65, 70]
        // Bottom row cols 2,3: pad with L3 = 70.
        // Then upward fill: each cell = the cell 2 cols left in the row below.
        let left = [10u8, 30, 50, 70];
        let mut dst = [0u8; 16];
        d207_predictor(&mut dst, 4, 4, &left);
        // Helper for readability with stride = 4.
        let at = |r: usize, c: usize| dst[r * 4 + c];
        // Col 0 across rows.
        assert_eq!(at(0, 0), 20);
        assert_eq!(at(1, 0), 40);
        assert_eq!(at(2, 0), 60);
        assert_eq!(at(3, 0), 70);
        // Col 1.
        assert_eq!(at(0, 1), 30);
        assert_eq!(at(1, 1), 50);
        assert_eq!(at(2, 1), 65);
        assert_eq!(at(3, 1), 70);
        // Bottom row cols 2 and 3 are 70.
        assert_eq!(at(3, 2), 70);
        assert_eq!(at(3, 3), 70);
        // Row 2 col 2 = row 3 col 0 = 70. Row 2 col 3 = row 3 col 1 = 70.
        assert_eq!(at(2, 2), 70);
        assert_eq!(at(2, 3), 70);
        // Row 1 col 2 = row 2 col 0 = 60. Row 1 col 3 = row 2 col 1 = 65.
        assert_eq!(at(1, 2), 60);
        assert_eq!(at(1, 3), 65);
        // Row 0 col 2 = row 1 col 0 = 40. Row 0 col 3 = row 1 col 1 = 50.
        assert_eq!(at(0, 2), 40);
        assert_eq!(at(0, 3), 50);
    }

    #[test]
    fn avg2_avg3_helpers() {
        // AVG2(0, 0) = 0
        // AVG2(0, 1) = 0 (since (0+1+1)>>1 = 1)
        // wait: (0+1+1) = 2, 2>>1 = 1. So = 1.
        assert_eq!(avg2(0, 0), 0);
        assert_eq!(avg2(0, 1), 1);
        assert_eq!(avg2(255, 255), 255);
        // AVG3 rounded-up average with center weighted 2.
        assert_eq!(avg3(0, 0, 0), 0);
        assert_eq!(avg3(10, 20, 30), 20); // (10+40+30+2)>>2 = 82>>2 = 20
        assert_eq!(avg3(255, 255, 255), 255);
    }

    #[test]
    fn predictors_work_at_bs_8_and_16() {
        // Sanity: V/H/DC at bs=8 and bs=16.
        for &bs in &[8usize, 16] {
            let above: Vec<u8> = (0..bs as u8).collect();
            let left: Vec<u8> = (0..bs as u8).map(|i| i * 2).collect();
            let mut dst = vec![0u8; bs * bs];

            v_predictor(&mut dst, bs, bs, PredCtx { above: &above, left: &left, above_left: 50 });
            // Row 0 == above.
            assert_eq!(&dst[0..bs], &above[..]);

            h_predictor(&mut dst, bs, bs, PredCtx { above: &above, left: &left, above_left: 50 });
            // Column 0 == left.
            for r in 0..bs {
                assert_eq!(dst[r * bs], left[r]);
            }

            dc_predictor(&mut dst, bs, bs, PredCtx { above: &above, left: &left, above_left: 50 });
            // DC: rounded average of 2*bs values.
            let sum: u32 = above.iter().chain(left.iter()).map(|&v| v as u32).sum();
            let dc = ((sum + bs as u32) / (2 * bs as u32)) as u8;
            assert!(dst.iter().all(|&v| v == dc));
        }
    }
}
