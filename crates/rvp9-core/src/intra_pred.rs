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
//! ## Not in this commit
//!
//! The 6 directional modes (`D45 / D63 / D117 / D135 / D153 / D207`)
//! — Task 5.2. They are sharper, weighted-interpolation filters with
//! distinct edge-handling per direction.
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
