// Wave A G11 — workspace lint `clippy::print_stderr` is "deny" for
// library crates. `predict_intra` carries an env-var-gated
// (`RVP9_TRACE_INTRA`) debug-dump scheduled for triage in the HLD
// §G11 deliverable 3 eprintln audit follow-up. Allow at the module
// level until that sweep lands.
#![allow(clippy::print_stderr)]

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
//! Task 5.2a added three directional intra modes — [`d45_predictor`],
//! [`d63_predictor`], [`d207_predictor`] — that use only `above` OR
//! only `left` neighbours. Task 5.2b adds the three corner
//! directionals — [`d117_predictor`], [`d135_predictor`],
//! [`d153_predictor`] — that consume `above + left + above_left`
//! simultaneously.
//!
//! Directional kernels (D45 / D63) read `above` pixels past the
//! current block (i.e. `above[bs..2*bs]` — the "above-right" pixels
//! of the block to the right of the current one). Callers must
//! populate a `2 * bs`-byte `above` slice when invoking those
//! predictors; if the right neighbour hasn't been decoded yet,
//! callers pad with the rightmost real `above` pixel (libvpx
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

/// VP9 intra-prediction modes. Discriminant values match libvpx's
/// `#define` values in `vp9_enums.h:110-119`.
///
/// Indexed by the 4-bit `y_mode` / `uv_mode` field from the entropy
/// decoder.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum IntraMode {
    /// DC_PRED — average of above + left neighbours.
    Dc = 0,
    /// V_PRED — copy `above` row.
    V = 1,
    /// H_PRED — broadcast `left[r]`.
    H = 2,
    /// D45_PRED — 45-degree downward-right edge.
    D45 = 3,
    /// D135_PRED — 135-degree edge (diagonal down-right from corner).
    D135 = 4,
    /// D117_PRED — 117-degree edge (close to vertical).
    D117 = 5,
    /// D153_PRED — 153-degree edge (close to horizontal).
    D153 = 6,
    /// D207_PRED — 207-degree edge (down-left).
    D207 = 7,
    /// D63_PRED — 63-degree downward-right edge (steeper than D45).
    D63 = 8,
    /// TM_PRED — true motion (clip(left[r] + above[c] - above_left)).
    Tm = 9,
}

/// Number of distinct VP9 intra modes (= `INTRA_MODES` in libvpx).
pub const NUM_INTRA_MODES: usize = 10;

impl IntraMode {
    /// Decode a raw 4-bit `y_mode` / `uv_mode` field. Returns `None`
    /// for values outside `0..=9`.
    #[inline]
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Dc),
            1 => Some(Self::V),
            2 => Some(Self::H),
            3 => Some(Self::D45),
            4 => Some(Self::D135),
            5 => Some(Self::D117),
            6 => Some(Self::D153),
            7 => Some(Self::D207),
            8 => Some(Self::D63),
            9 => Some(Self::Tm),
            _ => None,
        }
    }

    /// The hybrid-transform type ([`crate::transforms::TxType`]) used
    /// for intra blocks predicted with this mode. Mirrors libvpx
    /// `intra_mode_to_tx_type_lookup` (`vp9_reconintra.c:24-35`).
    ///
    /// This drives the decoder's selection of `iht4x4_16_add` /
    /// `iht8x8_64_add` / `iht16x16_256_add` and the matching
    /// scan-order from [`crate::transforms::scan_for`].
    #[inline]
    #[must_use]
    pub const fn tx_type(self) -> crate::transforms::TxType {
        use crate::transforms::TxType;
        match self {
            Self::Dc => TxType::DctDct,
            Self::V => TxType::AdstDct,
            Self::H => TxType::DctAdst,
            Self::D45 => TxType::DctDct,
            Self::D135 => TxType::AdstAdst,
            Self::D117 => TxType::AdstDct,
            Self::D153 => TxType::DctAdst,
            Self::D207 => TxType::DctAdst,
            Self::D63 => TxType::AdstDct,
            Self::Tm => TxType::AdstAdst,
        }
    }
}

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

/// D45 4×4 specialization — bit-for-bit port of libvpx
/// `vpx_d45_predictor_4x4_c` (`intrapred.c:346-365`).
///
/// libvpx defines `intra_pred_no_4x4(d45)` so the generic
/// [`d45_predictor`] is **only** invoked for 8×8 / 16×16 / 32×32; the
/// 4×4 path uses this hand-unrolled version which differs from the
/// generic. In particular `DST(3, 0) = AVG3(D, E, F)` rather than the
/// generic's `above_right = above[bs - 1] = D` (and analogous
/// differences for the other right-pad cells). Using the generic at
/// `bs == 4` produces a per-row constant offset that cascades through
/// downstream sub-8 intra-pred reads.
///
/// Reads `above[0..=7]` — caller must provide 2*bs = 8 valid samples.
pub fn d45_predictor_4x4(dst: &mut [u8], stride: usize, above: &[u8]) {
    debug_assert!(above.len() >= 8);
    let a = above[0];
    let b = above[1];
    let c = above[2];
    let d = above[3];
    let e = above[4];
    let f = above[5];
    let g = above[6];
    let h = above[7];

    // row 0
    dst[0] = avg3(a, b, c);
    dst[1] = avg3(b, c, d);
    dst[2] = avg3(c, d, e);
    dst[3] = avg3(d, e, f);
    // row 1
    dst[stride] = avg3(b, c, d);
    dst[stride + 1] = avg3(c, d, e);
    dst[stride + 2] = avg3(d, e, f);
    dst[stride + 3] = avg3(e, f, g);
    // row 2
    dst[2 * stride] = avg3(c, d, e);
    dst[2 * stride + 1] = avg3(d, e, f);
    dst[2 * stride + 2] = avg3(e, f, g);
    dst[2 * stride + 3] = avg3(f, g, h);
    // row 3
    dst[3 * stride] = avg3(d, e, f);
    dst[3 * stride + 1] = avg3(e, f, g);
    dst[3 * stride + 2] = avg3(f, g, h);
    dst[3 * stride + 3] = h;
}

/// D45 — predict using a 45-degree downward-right edge from the
/// top-row neighbours. Reads `above[0..=2*bs - 1]` (so `above` must
/// have at least `2 * bs` pixels). Does not use `left` or `above_left`.
///
/// Mirrors libvpx `d45_predictor` (`intrapred.c:62-79`). **Only used
/// for bs ∈ {8, 16, 32}** — for bs == 4, call [`d45_predictor_4x4`]
/// (libvpx `intra_pred_no_4x4(d45)` confirms the generic is not in the
/// 4x4 dispatch table).
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

/// D63 4×4 specialization — bit-for-bit port of libvpx
/// `vpx_d63_predictor_4x4_c` (`intrapred.c:299-320`).
///
/// libvpx defines `intra_pred_no_4x4(d63)` so the generic
/// [`d63_predictor`] is **only** invoked for 8×8 / 16×16 / 32×32; the
/// 4×4 path uses this hand-unrolled version which differs from the
/// generic. Critically, `DST(3, 2) = AVG2(E, F)` and
/// `DST(3, 3) = AVG3(E, F, G)`, **not** the generic's
/// `memset(..., above[bs - 1], ...)` = above[3] right-pad. Using the
/// generic at `bs == 4` shifts the bottom-right cells by an arbitrary
/// amount (up to ±5 typically) that then cascades through downstream
/// sub-8 H-mode reads of those cells via `left[r]`.
///
/// Reads `above[0..=6]` (A..G). Caller passes the 2*bs=8-byte above row.
pub fn d63_predictor_4x4(dst: &mut [u8], stride: usize, above: &[u8]) {
    debug_assert!(above.len() >= 7);
    let a = above[0];
    let b = above[1];
    let c = above[2];
    let d = above[3];
    let e = above[4];
    let f = above[5];
    let g = above[6];

    // Row 0 (DST(c, 0)).
    dst[0] = avg2(a, b);
    dst[1] = avg2(b, c);
    dst[2] = avg2(c, d);
    dst[3] = avg2(d, e);
    // Row 1 (DST(c, 1)).
    dst[stride] = avg3(a, b, c);
    dst[stride + 1] = avg3(b, c, d);
    dst[stride + 2] = avg3(c, d, e);
    dst[stride + 3] = avg3(d, e, f);
    // Row 2 (DST(c, 2)).
    dst[2 * stride] = avg2(b, c); // DST(0, 2) = DST(1, 0)
    dst[2 * stride + 1] = avg2(c, d); // DST(1, 2) = DST(2, 0)
    dst[2 * stride + 2] = avg2(d, e); // DST(2, 2) = DST(3, 0)
    dst[2 * stride + 3] = avg2(e, f); // DST(3, 2) — was above[bs-1] in generic
    // Row 3 (DST(c, 3)).
    dst[3 * stride] = avg3(b, c, d); // DST(0, 3) = DST(1, 1)
    dst[3 * stride + 1] = avg3(c, d, e); // DST(1, 3) = DST(2, 1)
    dst[3 * stride + 2] = avg3(d, e, f); // DST(2, 3) = DST(3, 1)
    dst[3 * stride + 3] = avg3(e, f, g); // DST(3, 3) — was above[bs-1] in generic
}

/// D63 — predict using a 63-degree downward-right edge. Reads
/// `above[0..=bs + 1]` minimum (the `c + 2 == bs + 1` boundary at
/// `c = bs - 1`); caller should supply `2 * bs` pixels for safety.
/// Does not use `left` or `above_left`.
///
/// Mirrors libvpx `d63_predictor` (`intrapred.c:45-60`). **Only used
/// for bs ∈ {8, 16, 32}** — for bs == 4, call [`d63_predictor_4x4`]
/// (libvpx `intra_pred_no_4x4(d63)` confirms the generic is not in the
/// 4x4 dispatch table).
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
                tail[..size].copy_from_slice(&head[copy_src_row_0..copy_src_row_0 + size]);
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

/// D117 — 117-degree edge (close to vertical, leaning right). Uses
/// `above[0..bs]`, `left[0..bs]`, and `above_left`.
///
/// Mirrors libvpx `d117_predictor` (`intrapred.c:81-104`). libvpx
/// reads `above[c - 1]` at `c == 0`, which is `above[-1]` —
/// equivalent to our `above_left`; we handle the boundary explicitly.
#[allow(clippy::needless_range_loop)]
pub fn d117_predictor(dst: &mut [u8], stride: usize, bs: usize, ctx: PredCtx<'_>) {
    debug_assert!(ctx.above.len() >= bs);
    debug_assert!(ctx.left.len() >= bs);
    let al = ctx.above_left;

    // Row 0: AVG2(above[c-1], above[c]) where above[-1] = above_left.
    dst[0] = avg2(al, ctx.above[0]);
    for c in 1..bs {
        dst[c] = avg2(ctx.above[c - 1], ctx.above[c]);
    }

    // Row 1: AVG3, first cell uses left[0] / above_left / above[0].
    let r1 = stride;
    dst[r1] = avg3(ctx.left[0], al, ctx.above[0]);
    // Row 1, c = 1: AVG3(above[-1], above[0], above[1]) — note above[-1] = al.
    if bs >= 2 {
        dst[r1 + 1] = avg3(al, ctx.above[0], ctx.above[1]);
    }
    for c in 2..bs {
        dst[r1 + c] = avg3(ctx.above[c - 2], ctx.above[c - 1], ctx.above[c]);
    }

    // First column for rows 2..bs (the libvpx pre-fill of the leading
    // column before the recursive copy block). Row 2 col 0 uses
    // (above_left, left[0], left[1]). Subsequent: AVG3 along left.
    if bs >= 3 {
        dst[2 * stride] = avg3(al, ctx.left[0], ctx.left[1]);
        for r in 3..bs {
            dst[r * stride] = avg3(ctx.left[r - 3], ctx.left[r - 2], ctx.left[r - 1]);
        }
    }

    // Remaining cells: dst[r][c] = dst[r-2][c-1] for r >= 2, c >= 1.
    for r in 2..bs {
        for c in 1..bs {
            dst[r * stride + c] = dst[(r - 2) * stride + c - 1];
        }
    }
}

/// D135 — 135-degree edge (diagonal down-right from top-left corner).
/// Uses `above[0..bs]`, `left[0..bs]`, and `above_left`.
///
/// Mirrors libvpx `d135_predictor` (`intrapred.c:106-132`). The
/// libvpx implementation precomputes a `bs + bs - 1` "border" running
/// from bottom-left through `above_left` to top-right, then each row
/// of `dst` is a `bs`-byte window into that border (shifting one cell
/// right per row).
#[allow(clippy::needless_range_loop)]
pub fn d135_predictor(dst: &mut [u8], stride: usize, bs: usize, ctx: PredCtx<'_>) {
    debug_assert!(ctx.above.len() >= bs);
    debug_assert!(ctx.left.len() >= bs);
    let al = ctx.above_left;

    // libvpx uses border[2*bs - 1] (= 32+32-1 = 63 max for 32x32).
    // We use a runtime-sized Vec since bs is dynamic; max 63 entries.
    let mut border = vec![0u8; 2 * bs - 1];

    // border[0..bs-2]: AVG3 along left, indexed bottom-up
    // (border[0] = AVG3(left[bs-3], left[bs-2], left[bs-1]); etc.)
    if bs >= 3 {
        for i in 0..bs - 2 {
            border[i] = avg3(
                ctx.left[bs - 3 - i],
                ctx.left[bs - 2 - i],
                ctx.left[bs - 1 - i],
            );
        }
    }
    // border[bs - 2] = AVG3(above_left, left[0], left[1]).
    if bs >= 2 {
        border[bs - 2] = avg3(al, ctx.left[0], ctx.left[1]);
    }
    // border[bs - 1] = AVG3(left[0], above_left, above[0]).
    border[bs - 1] = avg3(ctx.left[0], al, ctx.above[0]);
    // border[bs] = AVG3(above_left, above[0], above[1]).
    if bs >= 2 {
        border[bs] = avg3(al, ctx.above[0], ctx.above[1]);
    }
    // border[bs + 1..]: AVG3 along above, indexed ascending.
    if bs >= 3 {
        for i in 0..bs - 2 {
            border[bs + 1 + i] = avg3(ctx.above[i], ctx.above[i + 1], ctx.above[i + 2]);
        }
    }

    // Each row of dst is a bs-byte window into border, shifted right
    // by 1 per row: row i copies border[bs - 1 - i .. bs - 1 - i + bs].
    for i in 0..bs {
        let start = bs - 1 - i;
        dst[i * stride..i * stride + bs].copy_from_slice(&border[start..start + bs]);
    }
}

/// D153 — 153-degree edge (close to horizontal, leaning down). Uses
/// `above[0..bs]`, `left[0..bs]`, and `above_left`.
///
/// Mirrors libvpx `d153_predictor` (`intrapred.c:134-155`).
#[allow(clippy::needless_range_loop)]
pub fn d153_predictor(dst: &mut [u8], stride: usize, bs: usize, ctx: PredCtx<'_>) {
    debug_assert!(ctx.above.len() >= bs);
    debug_assert!(ctx.left.len() >= bs);
    let al = ctx.above_left;

    // Column 0 across all rows. Top cell: AVG2(above_left, left[0]).
    // Subsequent: AVG2 along left.
    dst[0] = avg2(al, ctx.left[0]);
    for r in 1..bs {
        dst[r * stride] = avg2(ctx.left[r - 1], ctx.left[r]);
    }

    // Column 1.
    if bs >= 2 {
        dst[1] = avg3(ctx.left[0], al, ctx.above[0]);
        dst[stride + 1] = avg3(al, ctx.left[0], ctx.left[1]);
        for r in 2..bs {
            dst[r * stride + 1] = avg3(ctx.left[r - 2], ctx.left[r - 1], ctx.left[r]);
        }
    }

    // Row 0 columns 2..bs: AVG3 along above (handling above[-1] for c=2).
    // libvpx: dst[c] = AVG3(above[c-1], above[c], above[c+1])
    //   with c starting at 0 of the (already +2'd) cursor.
    // We start c at 2 (real column index) and read above[c-2..=c].
    // Specifically: c=2 → AVG3(above[1], above[2], above[3])? No.
    // Re-reading libvpx :147-148:
    //   for c in 0..bs-2: dst[c] = AVG3(above[c-1], above[c], above[c+1])
    //   where the dst cursor has already been advanced 2 columns.
    // So at our column 2: AVG3(above[-1], above[0], above[1]) — uses al.
    // At our column 3: AVG3(above[0], above[1], above[2]).
    // At our column c (>= 2): AVG3(above[c-3], above[c-2], above[c-1])
    //   with above[-1] = al for the c=2 case.
    if bs >= 3 {
        // c = 2 boundary: above[c-3] = above[-1] = al.
        dst[2] = avg3(al, ctx.above[0], ctx.above[1]);
        for c in 3..bs {
            dst[c] = avg3(ctx.above[c - 3], ctx.above[c - 2], ctx.above[c - 1]);
        }
    }

    // Remaining cells: dst[r][c] = dst[r-1][c-2] for r >= 1, c >= 2.
    for r in 1..bs {
        for c in 2..bs {
            dst[r * stride + c] = dst[(r - 1) * stride + c - 2];
        }
    }
}

/// D207 4×4 specialization — bit-for-bit port of libvpx
/// `vpx_d207_predictor_4x4_c` (`intrapred.c:283-297`).
///
/// libvpx defines `intra_pred_no_4x4(d207)` so the generic
/// [`d207_predictor`] is **only** invoked for 8×8 / 16×16 / 32×32; the
/// 4×4 path uses this hand-unrolled version. Notably DST(3, 1) uses
/// `AVG3(K, L, L)` (rather than AVG3(left[bs-2], left[bs-1],
/// left[bs-1]) — these happen to coincide, but the placement of the
/// "saturated" cells differs from the generic walk).
///
/// Reads `left[0..=3]`.
pub fn d207_predictor_4x4(dst: &mut [u8], stride: usize, left: &[u8]) {
    debug_assert!(left.len() >= 4);
    let i = left[0];
    let j = left[1];
    let k = left[2];
    let l = left[3];

    // DST(0, 0) = AVG2(I, J)
    dst[0] = avg2(i, j);
    // DST(2, 0) = DST(0, 1) = AVG2(J, K)
    dst[2] = avg2(j, k);
    dst[stride] = avg2(j, k);
    // DST(2, 1) = DST(0, 2) = AVG2(K, L)
    dst[stride + 2] = avg2(k, l);
    dst[2 * stride] = avg2(k, l);
    // DST(1, 0) = AVG3(I, J, K)
    dst[1] = avg3(i, j, k);
    // DST(3, 0) = DST(1, 1) = AVG3(J, K, L)
    dst[3] = avg3(j, k, l);
    dst[stride + 1] = avg3(j, k, l);
    // DST(3, 1) = DST(1, 2) = AVG3(K, L, L)
    dst[stride + 3] = avg3(k, l, l);
    dst[2 * stride + 1] = avg3(k, l, l);
    // DST(3, 2) = DST(2, 2) = DST(0, 3) = DST(1, 3) = DST(2, 3) = DST(3, 3) = L
    dst[2 * stride + 2] = l;
    dst[2 * stride + 3] = l;
    dst[3 * stride] = l;
    dst[3 * stride + 1] = l;
    dst[3 * stride + 2] = l;
    dst[3 * stride + 3] = l;
}

/// D207 — predict using a 207-degree (i.e. 27-degree above horizontal,
/// downward-left) edge. Uses only `left`. Reads `left[0..bs]`.
///
/// Mirrors libvpx `d207_predictor` (`intrapred.c:21-43`). **Only used
/// for bs ∈ {8, 16, 32}** — for bs == 4, call [`d207_predictor_4x4`]
/// (libvpx `intra_pred_no_4x4(d207)` confirms the generic is not in
/// the 4x4 dispatch table).
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

// ============================================================================
// Dispatch
// ============================================================================

/// Single-entry dispatch for VP9 intra prediction. Given the
/// requested [`IntraMode`], the block size `bs`, the destination
/// buffer / stride, and a [`PredCtx`] capturing the available
/// neighbour pixels, calls the matching predictor.
///
/// For directional modes (`D45 / D63 / D117 / D135 / D153 / D207`)
/// `ctx.above` must have at least `2 * bs` valid bytes — the extra
/// `bs..2*bs` slots are the "above-right" pixels of the block to the
/// right of the current one. If unavailable, the caller pads with
/// the rightmost real `above` pixel per libvpx
/// `vp9_intra_predict_top_pixels` convention.
///
/// Mirrors libvpx's `pred[mode][txsz]` dispatch table at
/// `vp9_reconintra.c:351`.
///
/// `dc_avail` selects the DC variant when `mode == IntraMode::Dc`,
/// mirroring libvpx's `dc_pred[left_available][up_available][tx_size]`
/// dispatch at `vp9_reconintra.c:398`. For non-DC modes the flag is
/// ignored; libvpx's `build_intra_predictors` has already filled
/// `ctx.above` / `ctx.left` with the 127 / 129 unavailable-neighbour
/// values, so V / H / TM / directional kernels can be invoked
/// unconditionally on the materialised edges. DC is special because
/// libvpx swaps in a different kernel that averages only the
/// available neighbours instead of mixing in the 129 fill bytes.
#[inline]
pub fn intra_predict(
    mode: IntraMode,
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    ctx: PredCtx<'_>,
    dc_avail: DcAvailability,
    dispatch: &crate::dsp::SimdDispatch,
) {
    // Per libvpx `intra_pred_no_4x4(d{45,63,117,135,153,207})` the
    // bs == 4 path uses hand-unrolled specializations that differ from
    // the generic kernels at the bottom-right boundary. Most notably:
    //   D45  generic dst[bs-1] = above[bs-1]    (D, in 4x4 letter terms)
    //   D45  spec    DST(3, 0) = AVG3(D, E, F)  ← reads above[4..6]
    //   D63  generic right-pad memset uses above[bs-1] = D
    //   D63  spec    DST(3, 2) = AVG2(E, F), DST(3, 3) = AVG3(E, F, G)
    //   D207 generic and spec coincide except for an in-bound walk
    //         order; spec is explicit.
    // M.15 traced a +5 row-18 cols-28..31 cascade on
    // vp90-2-02-size-32x32.webm frame 1 to the D63 above-right pad
    // divergence: our generic produced row 2 col 3 = above[3] = 79
    // where libvpx's 4x4 spec produces AVG2(E, F) = AVG2(82, 85) = 84
    // — Δ=5 that propagates through downstream H-mode sub-1 reads.
    // The bs >= 8 path still uses the generic kernels which libvpx
    // also uses for those sizes (the macro is `intra_pred_no_4x4`,
    // not `intra_pred_only_4x4`).
    //
    // G4.3.1 — DC_PRED at bs=4 with both neighbours available routes
    // through the `dc_predictor_4x4` dispatch slot so AVX2 hosts run
    // the SIMD kernel. All other DC variants (DC_TOP / DC_LEFT /
    // DC_128 / bs > 4) stay on the scalar `dc_predict_with_availability`
    // path until a follow-up adds slots for them.
    let dispatched_dc_4x4 =
        mode == IntraMode::Dc && bs == 4 && dc_avail.above_available && dc_avail.left_available;
    if dispatched_dc_4x4 {
        (dispatch.dc_predictor_4x4())(dst, stride, ctx.above, ctx.left);
    } else {
        match (mode, bs == 4) {
            (IntraMode::Dc, _) => {
                dc_predict_with_availability(
                    dst, stride, bs, ctx.above, ctx.left, dc_avail, dispatch,
                );
            }
            (IntraMode::V, _) => (dispatch.v_predictor())(dst, stride, bs, ctx.above),
            (IntraMode::H, _) => (dispatch.h_predictor())(dst, stride, bs, ctx.left),
            // Wave K — directional modes route through dispatch slots
            // (AVX2 on capable x86_64 hosts, scalar elsewhere). The
            // slot fns route the `bs == 4` specialization internally for
            // D45/D63/D207, so no `bs == 4` arm is needed here.
            (IntraMode::D45, _) => (dispatch.d45_predictor())(dst, stride, bs, ctx.above),
            (IntraMode::D135, _) => {
                (dispatch.d135_predictor())(dst, stride, bs, ctx.above, ctx.left, ctx.above_left)
            }
            (IntraMode::D117, _) => {
                (dispatch.d117_predictor())(dst, stride, bs, ctx.above, ctx.left, ctx.above_left)
            }
            (IntraMode::D153, _) => {
                (dispatch.d153_predictor())(dst, stride, bs, ctx.above, ctx.left, ctx.above_left)
            }
            (IntraMode::D207, _) => (dispatch.d207_predictor())(dst, stride, bs, ctx.left),
            (IntraMode::D63, _) => (dispatch.d63_predictor())(dst, stride, bs, ctx.above),
            (IntraMode::Tm, _) => {
                (dispatch.tm_predictor())(dst, stride, bs, ctx.above, ctx.left, ctx.above_left)
            }
        }
    }

    // Stage 3 stretch M.9 — env-gated trace. Dumps mode + edge inputs +
    // prediction output for blocks matching the filter. Used to localize
    // post-iht-fix divergences (e.g. 32×32 keyframe at col=4..7 row=0..3).
    // Filter is hard-coded by env var content:
    //   RVP9_TRACE_INTRA=all       — every call (very noisy)
    //   RVP9_TRACE_INTRA=4x4       — only 4x4 calls
    //   (unset → no trace)
    if let Some(filter) = crate::trace_flags::trace_flags().intra.as_deref() {
        trace_intra_pred(filter, mode, bs, ctx, dst, stride);
    }
}

/// Cold-path dump for the `RVP9_TRACE_INTRA` debug filter, factored out
/// of [`intra_predict`] so the hot predictor body stays small enough for
/// LTO to inline into the per-TX reconstruction loop (intra recon is the
/// dominant decode cost — see
/// `notes/2026.05.29 - NOTE - decoder perf intra diagnosis.md`).
/// Behaviour is byte-identical to the inline form: same filter, same
/// dump, same output.
#[cold]
#[inline(never)]
fn trace_intra_pred(
    filter: &str,
    mode: IntraMode,
    bs: usize,
    ctx: PredCtx<'_>,
    dst: &[u8],
    stride: usize,
) {
    let want = filter == "all" || (filter == "4x4" && bs == 4);
    if !want {
        return;
    }
    eprint!(
        "RVP9_TRACE_INTRA mode={mode:?} bs={bs} above_left={} above=[",
        ctx.above_left
    );
    for (i, &v) in ctx.above.iter().take(2 * bs).enumerate() {
        if i > 0 {
            eprint!(",");
        }
        eprint!("{v}");
    }
    eprint!("] left=[");
    for (i, &v) in ctx.left.iter().take(bs).enumerate() {
        if i > 0 {
            eprint!(",");
        }
        eprint!("{v}");
    }
    eprint!("] pred=[");
    for r in 0..bs {
        if r > 0 {
            eprint!(";");
        }
        for c in 0..bs {
            if c > 0 {
                eprint!(",");
            }
            eprint!("{}", dst[r * stride + c]);
        }
    }
    eprintln!("]");
}

/// Neighbour-availability hint for the DC predictor family. The
/// decoder sets `above_available` / `left_available` based on whether
/// the corresponding block has already been decoded (and isn't at
/// the picture boundary); the dispatcher then picks the appropriate
/// `dc_*_predictor` variant.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DcAvailability {
    /// Whether the row above the current block is valid.
    pub above_available: bool,
    /// Whether the column to the left is valid.
    pub left_available: bool,
}

/// DC-predictor dispatch with neighbour-availability fallback. Picks
/// between [`dc_predictor`] (both available), [`dc_top_predictor`]
/// (only above), [`dc_left_predictor`] (only left), and
/// [`dc_128_predictor`] (neither). Mirrors libvpx's `dc_pred[][]`
/// table at `vp9_reconintra.c:330-348`.
///
/// Wave I I4 — routes through the per-arm `SimdDispatch` slots so
/// AVX2 hosts get the lane-parallel kernels. The legacy
/// `dispatch.dc_predictor_4x4` slot still handles the
/// `(bs = 4, both-avail)` fast path (called from `intra_predict`);
/// this router covers every other DC variant including bs ∈ {8, 16, 32}
/// with both neighbours available and all four arms for any bs when at
/// least one neighbour is missing.
pub fn dc_predict_with_availability(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    availability: DcAvailability,
    dispatch: &crate::dsp::SimdDispatch,
) {
    match (availability.above_available, availability.left_available) {
        (true, true) => (dispatch.dc_predict_full())(dst, stride, bs, above, left),
        (true, false) => (dispatch.dc_predict_top())(dst, stride, bs, above),
        (false, true) => (dispatch.dc_predict_left())(dst, stride, bs, left),
        (false, false) => (dispatch.dc_predict_128())(dst, stride, bs),
    }
}

// ============================================================================
// Edge-builder
// ============================================================================

/// Maximum block side (`Tx32x32`) — used to size [`EdgeBuffers`].
const MAX_BS: usize = 32;

/// Sentinel for missing-above neighbours. Mirrors libvpx's `127` fill
/// at `vp9_reconintra.c:345-346, 391-392`.
pub const ABOVE_UNAVAILABLE_FILL: u8 = 127;

/// Sentinel for missing-left neighbours. Mirrors libvpx's `129` fill
/// at `vp9_reconintra.c:318, 343, 389`.
pub const LEFT_UNAVAILABLE_FILL: u8 = 129;

/// Scratch storage for the intra-pred edge-builder.
///
/// Mirrors libvpx's two stack-local arrays at
/// `vp9_reconintra.c:270-272`:
///
/// ```c
/// DECLARE_ALIGNED(16, uint8_t, left_col[32]);
/// DECLARE_ALIGNED(16, uint8_t, above_data[64 + 16]);
/// uint8_t *above_row = above_data + 16;
/// ```
///
/// The combined layout matches libvpx so [`build_intra_predictors`]
/// can write `above_row[-1]` (the top-left corner) into the byte at
/// `above_data[15]`. The kernels consume `above` as a slice starting
/// at `above_data[16]`, with up to `2 * bs == 64` valid bytes (D45 /
/// D63 above-right extension).
///
/// Callers materialise an `EdgeBuffers` on the stack, hand it to
/// [`build_intra_predictors`], and convert to a [`PredCtx`] via
/// [`EdgeBuffers::pred_ctx`] for [`intra_predict`].
#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct EdgeBuffers {
    /// 16 bytes of leading padding; byte `[15]` holds `above_left`
    /// (libvpx's `above_row[-1]`). Bytes `[16..16 + 2*bs]` hold the
    /// above row plus above-right extension.
    pub above_data: [u8; MAX_BS * 2 + 16],
    /// Left column. `bs` entries are written; trailing bytes are
    /// untouched.
    pub left_col: [u8; MAX_BS],
}

impl Default for EdgeBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgeBuffers {
    /// Allocate a zeroed pair of scratch buffers.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            above_data: [0u8; MAX_BS * 2 + 16],
            left_col: [0u8; MAX_BS],
        }
    }

    /// Borrow this buffer as a [`PredCtx`] for `bs`-pixel blocks. The
    /// returned `above` slice spans `2 * bs` bytes (sufficient for the
    /// `D45 / D63` above-right consumers); the `left` slice spans
    /// `bs` bytes; `above_left` is read from `above_data[15]`.
    #[inline]
    #[must_use]
    pub fn pred_ctx(&self, bs: usize) -> PredCtx<'_> {
        debug_assert!(bs <= MAX_BS);
        PredCtx {
            above: &self.above_data[16..16 + 2 * bs],
            left: &self.left_col[..bs],
            above_left: self.above_data[15],
        }
    }
}

/// Mode → neighbour-need bitmask. Mirrors libvpx's `extend_modes`
/// table at `vp9_reconintra.c:43-54`.
const NEED_LEFT: u8 = 1 << 1;
const NEED_ABOVE: u8 = 1 << 2;
const NEED_ABOVERIGHT: u8 = 1 << 3;

const fn extend_mask(mode: IntraMode) -> u8 {
    match mode {
        IntraMode::Dc => NEED_ABOVE | NEED_LEFT,
        IntraMode::V => NEED_ABOVE,
        IntraMode::H => NEED_LEFT,
        IntraMode::D45 => NEED_ABOVERIGHT,
        IntraMode::D135 => NEED_LEFT | NEED_ABOVE,
        IntraMode::D117 => NEED_LEFT | NEED_ABOVE,
        IntraMode::D153 => NEED_LEFT | NEED_ABOVE,
        IntraMode::D207 => NEED_LEFT,
        IntraMode::D63 => NEED_ABOVERIGHT,
        IntraMode::Tm => NEED_LEFT | NEED_ABOVE,
    }
}

/// Populate `edges` (above / left / above-left) from already-reconstructed
/// pixels in `dst_plane`, applying the 127 / 129 unavailable-neighbour
/// rules and the directional-mode `2 * bs` above-right extension.
///
/// Mirrors libvpx `build_intra_predictors` at
/// `vp9/common/vp9_reconintra.c:263-403` for the 8-bit (non-HBD) path.
///
/// # Parameters
///
/// - `dst_plane`: read-only view of the destination plane. Must cover
///   at least `dst_stride * (px_y + bs)` bytes (or `px_y` rows if
///   `have_top` is false). `dst_plane[(px_y - 1) * dst_stride .. ]`
///   is the row above; `dst_plane[r * dst_stride + (px_x - 1)]` is the
///   leftward column.
/// - `dst_stride`: plane stride in bytes.
/// - `px_x`, `px_y`: top-left of the block being predicted, in
///   plane-sample units.
/// - `plane_width`, `plane_height`: plane dimensions in the same units
///   (used to derive `right_clipped` / `bottom_clipped` per libvpx
///   `vp9_reconintra.c:286-297, 302, 326, 354`).
/// - `tx_size`: the transform size of the block.
/// - `mode`: the prediction mode; selects which edges are populated
///   via [`extend_mask`].
/// - `have_top`: whether the row above is in-frame **and** already
///   decoded. Mirrors libvpx `up_available` (`vp9_reconintra.c:411`).
/// - `have_left`: whether the column to the left is in-frame and
///   already decoded. Mirrors libvpx `left_available`.
/// - `have_right`: whether the **top-right** neighbour TX block is
///   in-frame and decoded — used by libvpx only for the `bs == 4`
///   above-right fast path (`vp9_reconintra.c:337, 357-358, 365,
///   379-385`). For TX blocks with `bs > 4` this flag is unused.
/// - `edges`: scratch buffer overwritten by this call.
///
/// After the call, [`EdgeBuffers::pred_ctx`] returns a [`PredCtx`]
/// ready for [`intra_predict`].
#[allow(clippy::too_many_arguments)] // mirrors libvpx 1:1
pub fn build_intra_predictors(
    dst_plane: &[u8],
    dst_stride: usize,
    px_x: usize,
    px_y: usize,
    plane_width: usize,
    plane_height: usize,
    tx_size: crate::transforms::TxSize,
    mode: IntraMode,
    have_top: bool,
    have_left: bool,
    have_right: bool,
    edges: &mut EdgeBuffers,
) {
    let bs = tx_size.side();
    debug_assert!(bs <= MAX_BS);
    debug_assert!(px_x < plane_width, "block top-left must be in-frame (x)");
    debug_assert!(px_y < plane_height, "block top-left must be in-frame (y)");

    let mask = extend_mask(mode);

    // NEED_LEFT — mirrors vp9_reconintra.c:299-320.
    if (mask & NEED_LEFT) != 0 {
        if have_left {
            // `ref = dst_plane[px_y * dst_stride + px_x]`; the leftward
            // column we read is at `ref + i * ref_stride - 1`.
            let base = px_y * dst_stride + px_x;
            // Slow path if the block extends past the frame bottom
            // (libvpx checks `xd->mb_to_bottom_edge < 0`, equivalent).
            let extend_bottom_full = px_y + bs <= plane_height;
            if extend_bottom_full {
                // Hot interior-block path (this loop runs per intra TX
                // sub-block — the dominant decode cost). Bound-check the
                // strided read span ONCE via a sub-slice, then walk it
                // with `step_by` (internally unchecked) and zip into
                // `left_col[..bs]` (checked once). Elides the per-element
                // bounds checks of the prior `for i in 0..bs` form.
                // Byte-identical: same reads, same writes, same order.
                let first = base - 1;
                let last = first + (bs - 1) * dst_stride;
                let span = &dst_plane[first..=last];
                for (slot, &px) in edges.left_col[..bs]
                    .iter_mut()
                    .zip(span.iter().step_by(dst_stride))
                {
                    *slot = px;
                }
            } else {
                let extend_bottom = plane_height - px_y;
                for i in 0..extend_bottom {
                    edges.left_col[i] = dst_plane[base + i * dst_stride - 1];
                }
                // Replicate the last in-frame sample down. Mirrors
                // vp9_reconintra.c:310-311.
                let last = dst_plane[base + (extend_bottom - 1) * dst_stride - 1];
                for i in extend_bottom..bs {
                    edges.left_col[i] = last;
                }
            }
        } else {
            edges.left_col[..bs].fill(LEFT_UNAVAILABLE_FILL);
        }
    }

    // NEED_ABOVE — mirrors vp9_reconintra.c:322-348.
    if (mask & NEED_ABOVE) != 0 {
        if have_top {
            let above_ref_start = (px_y - 1) * dst_stride + px_x;
            let right_clipped = px_x + bs > plane_width;
            if right_clipped {
                // Slow path: copy what's in-frame and replicate the
                // last in-frame sample to fill `bs`.
                // libvpx vp9_reconintra.c:328-334. Note libvpx skips
                // the write entirely when `px_x > plane_width`, leaving
                // `above_row` uninitialised; per LLD §8.3 the partition
                // driver never asks intra-pred to predict a block whose
                // top-left is outside the frame, so `px_x <= plane_width`
                // holds.
                let r = plane_width - px_x; // in [1, bs)
                edges.above_data[16..16 + r]
                    .copy_from_slice(&dst_plane[above_ref_start..above_ref_start + r]);
                let last = edges.above_data[16 + r - 1];
                edges.above_data[16 + r..16 + bs].fill(last);
            } else {
                edges.above_data[16..16 + bs]
                    .copy_from_slice(&dst_plane[above_ref_start..above_ref_start + bs]);
            }
            // above_row[-1] — vp9_reconintra.c:343.
            edges.above_data[15] = if have_left {
                dst_plane[above_ref_start - 1]
            } else {
                LEFT_UNAVAILABLE_FILL
            };
        } else {
            edges.above_data[16..16 + bs].fill(ABOVE_UNAVAILABLE_FILL);
            edges.above_data[15] = ABOVE_UNAVAILABLE_FILL;
        }
    }

    // NEED_ABOVERIGHT — mirrors vp9_reconintra.c:350-394.
    if (mask & NEED_ABOVERIGHT) != 0 {
        if have_top {
            let above_ref_start = (px_y - 1) * dst_stride + px_x;
            let right_clipped = px_x + bs > plane_width;
            if right_clipped {
                // Three sub-cases per libvpx vp9_reconintra.c:355-376.
                if px_x + 2 * bs <= plane_width {
                    // Full 2*bs above pixels are in-frame.
                    if have_right && bs == 4 {
                        edges.above_data[16..16 + 2 * bs]
                            .copy_from_slice(&dst_plane[above_ref_start..above_ref_start + 2 * bs]);
                    } else {
                        edges.above_data[16..16 + bs]
                            .copy_from_slice(&dst_plane[above_ref_start..above_ref_start + bs]);
                        let last = edges.above_data[16 + bs - 1];
                        edges.above_data[16 + bs..16 + 2 * bs].fill(last);
                    }
                } else if px_x + bs <= plane_width {
                    // `bs` pixels in-frame; replicate to fill 2*bs.
                    let r = plane_width - px_x; // in [bs, 2*bs)
                    if have_right && bs == 4 {
                        edges.above_data[16..16 + r]
                            .copy_from_slice(&dst_plane[above_ref_start..above_ref_start + r]);
                        let last = edges.above_data[16 + r - 1];
                        edges.above_data[16 + r..16 + 2 * bs].fill(last);
                    } else {
                        edges.above_data[16..16 + bs]
                            .copy_from_slice(&dst_plane[above_ref_start..above_ref_start + bs]);
                        let last = edges.above_data[16 + bs - 1];
                        edges.above_data[16 + bs..16 + 2 * bs].fill(last);
                    }
                } else {
                    // Fewer than `bs` pixels in-frame.
                    let r = plane_width - px_x; // in [1, bs)
                    edges.above_data[16..16 + r]
                        .copy_from_slice(&dst_plane[above_ref_start..above_ref_start + r]);
                    let last = edges.above_data[16 + r - 1];
                    edges.above_data[16 + r..16 + 2 * bs].fill(last);
                }
            } else {
                // Fast path: 2*bs above pixels available without clip.
                edges.above_data[16..16 + bs]
                    .copy_from_slice(&dst_plane[above_ref_start..above_ref_start + bs]);
                if bs == 4 && have_right {
                    edges.above_data[16 + bs..16 + 2 * bs].copy_from_slice(
                        &dst_plane[above_ref_start + bs..above_ref_start + 2 * bs],
                    );
                } else {
                    let last = edges.above_data[16 + bs - 1];
                    edges.above_data[16 + bs..16 + 2 * bs].fill(last);
                }
            }
            // above_row[-1] — vp9_reconintra.c:389.
            edges.above_data[15] = if have_left {
                dst_plane[above_ref_start - 1]
            } else {
                LEFT_UNAVAILABLE_FILL
            };
        } else {
            edges.above_data[16..16 + 2 * bs].fill(ABOVE_UNAVAILABLE_FILL);
            edges.above_data[15] = ABOVE_UNAVAILABLE_FILL;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::caps::Caps;
    use crate::dsp::dispatch::SimdDispatch;

    fn make_dst(bs: usize, stride: usize) -> Vec<u8> {
        vec![0u8; stride * bs]
    }

    /// G4.3.1 — locally-built scalar-only dispatch table for tests
    /// that care only about the legacy scalar dispatch behaviour
    /// (not the SIMD slot itself). Forcing scalar caps means these
    /// tests behave identically on AVX2 and non-AVX2 hosts.
    fn scalar_dispatch() -> SimdDispatch {
        SimdDispatch::build(Caps::default())
    }

    #[test]
    fn intra_mode_discriminants_match_libvpx() {
        // libvpx vp9_enums.h:110-119
        assert_eq!(IntraMode::Dc as u8, 0);
        assert_eq!(IntraMode::V as u8, 1);
        assert_eq!(IntraMode::H as u8, 2);
        assert_eq!(IntraMode::D45 as u8, 3);
        assert_eq!(IntraMode::D135 as u8, 4);
        assert_eq!(IntraMode::D117 as u8, 5);
        assert_eq!(IntraMode::D153 as u8, 6);
        assert_eq!(IntraMode::D207 as u8, 7);
        assert_eq!(IntraMode::D63 as u8, 8);
        assert_eq!(IntraMode::Tm as u8, 9);
    }

    #[test]
    fn intra_mode_from_u8_round_trip() {
        for v in 0u8..=9 {
            let m = IntraMode::from_u8(v).expect("0..=9 valid");
            assert_eq!(m as u8, v);
        }
        assert_eq!(IntraMode::from_u8(10), None);
        assert_eq!(IntraMode::from_u8(255), None);
    }

    #[test]
    fn intra_mode_tx_type_lookup_matches_libvpx() {
        // Mirrors libvpx vp9_reconintra.c:24-35 exactly.
        use crate::transforms::TxType;
        assert_eq!(IntraMode::Dc.tx_type(), TxType::DctDct);
        assert_eq!(IntraMode::V.tx_type(), TxType::AdstDct);
        assert_eq!(IntraMode::H.tx_type(), TxType::DctAdst);
        assert_eq!(IntraMode::D45.tx_type(), TxType::DctDct);
        assert_eq!(IntraMode::D135.tx_type(), TxType::AdstAdst);
        assert_eq!(IntraMode::D117.tx_type(), TxType::AdstDct);
        assert_eq!(IntraMode::D153.tx_type(), TxType::DctAdst);
        assert_eq!(IntraMode::D207.tx_type(), TxType::DctAdst);
        assert_eq!(IntraMode::D63.tx_type(), TxType::AdstDct);
        assert_eq!(IntraMode::Tm.tx_type(), TxType::AdstAdst);
    }

    /// `NUM_INTRA_MODES` matches the enum's variant count.
    #[test]
    fn num_intra_modes_constant() {
        assert_eq!(NUM_INTRA_MODES, 10);
    }

    // ---- Dispatch ----

    /// `intra_predict` routes to the right kernel for each mode.
    /// We compare its output to a direct kernel call for the same
    /// inputs.
    #[test]
    fn intra_predict_dispatch_matches_direct_call_all_modes() {
        // 4×4 block with distinguishable above / left / above_left.
        let above = [10u8, 20, 30, 40, 50, 60, 70, 80]; // 2*bs for directional
        let left = [100u8, 110, 120, 130];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 50,
        };

        for raw_mode in 0u8..=9 {
            let mode = IntraMode::from_u8(raw_mode).unwrap();

            // Dispatched call. With both neighbours available the DC arm
            // is identical to the legacy `dc_predictor`.
            let mut dst_disp = [0u8; 16];
            intra_predict(
                mode,
                &mut dst_disp,
                4,
                4,
                ctx,
                DcAvailability {
                    above_available: true,
                    left_available: true,
                },
                &scalar_dispatch(),
            );

            // Direct kernel call. The bs == 4 dispatch goes through the
            // libvpx 4x4 specialization (M.15), so the direct comparison
            // for D45 / D63 / D207 must invoke the same specialization.
            let mut dst_direct = [0u8; 16];
            match mode {
                IntraMode::Dc => dc_predictor(&mut dst_direct, 4, 4, ctx),
                IntraMode::V => v_predictor(&mut dst_direct, 4, 4, ctx),
                IntraMode::H => h_predictor(&mut dst_direct, 4, 4, ctx),
                IntraMode::D45 => d45_predictor_4x4(&mut dst_direct, 4, &above),
                IntraMode::D135 => d135_predictor(&mut dst_direct, 4, 4, ctx),
                IntraMode::D117 => d117_predictor(&mut dst_direct, 4, 4, ctx),
                IntraMode::D153 => d153_predictor(&mut dst_direct, 4, 4, ctx),
                IntraMode::D207 => d207_predictor_4x4(&mut dst_direct, 4, &left),
                IntraMode::D63 => d63_predictor_4x4(&mut dst_direct, 4, &above),
                IntraMode::Tm => tm_predictor(&mut dst_direct, 4, 4, ctx),
            }

            assert_eq!(dst_disp, dst_direct, "mode {mode:?} dispatch mismatch");
        }
    }

    /// G4.3.1 — the `dc_predictor_4x4` dispatch slot fires for DC_PRED
    /// at `bs = 4` with both neighbours available, and the AVX2 path
    /// (when caps.avx2 == true) produces byte-identical output to the
    /// scalar reference path.
    ///
    /// This is the production-wiring oracle: feeding identical inputs
    /// through an AVX2-enabled SimdDispatch and a forced-scalar one
    /// MUST yield equal `dst` buffers. On non-AVX2 hosts (`caps.avx2`
    /// initially false) both dispatches land on the scalar adapter and
    /// the test degenerates to scalar-vs-scalar — still a valid
    /// regression against the dispatch threading itself.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dispatch_dc_pred_4x4_avx2_matches_scalar() {
        use crate::dsp::caps::{Caps, detect};

        // Distinguishable neighbour samples so any per-lane bug shows up.
        let above = [10u8, 25, 80, 200, 17, 42, 63, 99];
        let left = [33u8, 77, 128, 254];
        let ctx = PredCtx {
            above: &above[..4],
            left: &left,
            above_left: 50,
        };
        let dc_avail = DcAvailability {
            above_available: true,
            left_available: true,
        };

        // Scalar dispatch — caps.avx2 forced off.
        let mut scalar_caps = detect();
        // On non-x86_64 the `avx2` field doesn't exist; this whole
        // test is gated on x86_64.
        scalar_caps.avx2 = false;
        let scalar_d = SimdDispatch::build(scalar_caps);

        // Native dispatch — whatever the CPU actually supports. When
        // the test runs on an AVX2 host this lands on the AVX2 kernel.
        let native_d = SimdDispatch::build(Caps::detect());

        let mut dst_scalar = [0u8; 16];
        intra_predict(
            IntraMode::Dc,
            &mut dst_scalar,
            4,
            4,
            ctx,
            dc_avail,
            &scalar_d,
        );

        let mut dst_native = [0u8; 16];
        intra_predict(
            IntraMode::Dc,
            &mut dst_native,
            4,
            4,
            ctx,
            dc_avail,
            &native_d,
        );

        assert_eq!(
            dst_native, dst_scalar,
            "AVX2 dc_predictor_4x4 must agree with scalar reference \
             byte-for-byte through `intra_predict` (native caps = {:?})",
            native_d.caps,
        );

        // Sanity: when the test host has avx2, the two dispatches do
        // hold *different* function pointers — that's what guarantees
        // we actually exercised the AVX2 path and not just compared
        // scalar-to-scalar.
        if native_d.caps.avx2 {
            assert_ne!(
                native_d.dc_predictor_4x4() as usize,
                scalar_d.dc_predictor_4x4() as usize,
                "with avx2 caps the dispatch slot must hold the AVX2 \
                 kernel, not the scalar adapter — if these match the \
                 test is not exercising the AVX2 path",
            );
        }
    }

    /// DC availability dispatch: 4 combinations of
    /// `(above_available, left_available)` route to the right kernel.
    /// Wave I I4 — now takes a `&SimdDispatch` argument; uses the
    /// crate-test `scalar_dispatch()` helper to force scalar caps so
    /// the expected DC byte is computable by hand.
    #[test]
    fn dc_predict_with_availability_picks_right_kernel() {
        let above = [16u8, 16, 16, 16];
        let left = [32u8, 32, 32, 32];
        let dispatch = scalar_dispatch();

        // Both available → dc_predictor: (16*4 + 32*4 + 4) / 8 = 24.
        let mut dst = [0u8; 16];
        dc_predict_with_availability(
            &mut dst,
            4,
            4,
            &above,
            &left,
            DcAvailability {
                above_available: true,
                left_available: true,
            },
            &dispatch,
        );
        assert!(dst.iter().all(|&v| v == 24));

        // Only above → dc_top_predictor: (16*4 + 2) / 4 = 16.
        let mut dst = [0u8; 16];
        dc_predict_with_availability(
            &mut dst,
            4,
            4,
            &above,
            &left,
            DcAvailability {
                above_available: true,
                left_available: false,
            },
            &dispatch,
        );
        assert!(dst.iter().all(|&v| v == 16));

        // Only left → dc_left_predictor: (32*4 + 2) / 4 = 32.
        let mut dst = [0u8; 16];
        dc_predict_with_availability(
            &mut dst,
            4,
            4,
            &above,
            &left,
            DcAvailability {
                above_available: false,
                left_available: true,
            },
            &dispatch,
        );
        assert!(dst.iter().all(|&v| v == 32));

        // Neither → dc_128_predictor: 128.
        let mut dst = [0u8; 16];
        dc_predict_with_availability(
            &mut dst,
            4,
            4,
            &above,
            &left,
            DcAvailability {
                above_available: false,
                left_available: false,
            },
            &dispatch,
        );
        assert!(dst.iter().all(|&v| v == 128));
    }

    /// `intra_predict(Dc, ..., dc_avail)` routes to the correct
    /// `dc_*_predictor` variant per libvpx
    /// `dc_pred[left_available][up_available]`
    /// (`vp9_reconintra.c:398`). Regression for Stage 3 stretch M.12 —
    /// the unconditional `dc_predictor` dispatch produced a +1 bias on
    /// frame-edge sub-8 blocks (32×32 vector frame 1, sub-0 at
    /// `mi_row=1, mi_col=0`, plane Y) because the `left` slice was
    /// pre-filled with `LEFT_UNAVAILABLE_FILL = 129` and dragged the
    /// DC average up.
    #[test]
    fn intra_predict_dc_routes_via_availability() {
        let above = [16u8, 16, 16, 16];
        // libvpx `build_intra_predictors` fills the left col with
        // `LEFT_UNAVAILABLE_FILL = 129` when `left_available == 0`.
        // The legacy `dc_predictor` would mix that 129 into the
        // average; `dc_top_predictor` ignores it.
        let left = [129u8, 129, 129, 129];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 129,
        };

        let dispatch = scalar_dispatch();

        // (above_only) → dc_top_predictor: (16*4 + 2) / 4 = 16.
        let mut dst = [0u8; 16];
        intra_predict(
            IntraMode::Dc,
            &mut dst,
            4,
            4,
            ctx,
            DcAvailability {
                above_available: true,
                left_available: false,
            },
            &dispatch,
        );
        assert!(
            dst.iter().all(|&v| v == 16),
            "above-only → 16; mixing left=129 would give 73",
        );

        // (left_only) → dc_left_predictor: (129*4 + 2) / 4 = 129.
        let mut dst = [0u8; 16];
        intra_predict(
            IntraMode::Dc,
            &mut dst,
            4,
            4,
            ctx,
            DcAvailability {
                above_available: false,
                left_available: true,
            },
            &dispatch,
        );
        assert!(dst.iter().all(|&v| v == 129));

        // (neither) → dc_128_predictor: 128.
        let mut dst = [0u8; 16];
        intra_predict(
            IntraMode::Dc,
            &mut dst,
            4,
            4,
            ctx,
            DcAvailability {
                above_available: false,
                left_available: false,
            },
            &dispatch,
        );
        assert!(dst.iter().all(|&v| v == 128));

        // (both) → dc_predictor: (16*4 + 129*4 + 4) / 8 = 584/8 = 73.
        let mut dst = [0u8; 16];
        intra_predict(
            IntraMode::Dc,
            &mut dst,
            4,
            4,
            ctx,
            DcAvailability {
                above_available: true,
                left_available: true,
            },
            &dispatch,
        );
        assert!(dst.iter().all(|&v| v == 73));
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

    /// Stage 3 stretch M.15 — load-bearing differential. The libvpx 4x4
    /// specializations (`vpx_d{45,63,207}_predictor_4x4_c`) differ from
    /// the generic kernels at the bottom-right boundary; on
    /// `vp90-2-02-size-32x32.webm` frame 1 the M.15 oracle pinned the
    /// pre-LF row-18 cols-28..31 +5 uniform Δ to D63 sub-0's
    /// `DST(3, 2) = AVG2(E, F)` cell — generic produced
    /// `above[bs-1] = above[3] = 79` where the spec produces
    /// `AVG2(82, 85) = 84`.
    #[test]
    fn d63_predictor_4x4_m15_smoking_gun() {
        // Captured trace: vp90-2-02-size-32x32.webm frame 1 block
        // (24, 16) sub-0 D63 inputs.
        let above = [93u8, 112, 131, 79, 82, 85, 87, 88];
        let mut dst = [0u8; 16];
        d63_predictor_4x4(&mut dst, 4, &above);

        // libvpx vpx_d63_predictor_4x4_c output (verified by extending
        // tools/libvpx_decode_coefs_oracle/oracle.c with an M.15 probe
        // that calls the libvpx-stock predictor on these exact inputs):
        //   row 0: [103, 122, 105,  81]
        //   row 1: [112, 113,  93,  82]
        //   row 2: [122, 105,  81,  84]  ← col 3 = AVG2(E, F) = AVG2(82, 85) = 84
        //   row 3: [113,  93,  82,  85]
        let expected: [u8; 16] = [
            103, 122, 105, 81, // row 0
            112, 113, 93, 82, // row 1
            122, 105, 81, 84, // row 2 — load-bearing col 3
            113, 93, 82, 85, // row 3
        ];
        assert_eq!(dst, expected);
    }

    /// Stage 3 stretch M.15 — D45 4x4 specialization.
    /// libvpx `vpx_d45_predictor_4x4_c` uses `DST(3, 0) = AVG3(D, E, F)`
    /// rather than the generic's right-pad = `above[bs - 1] = D`.
    #[test]
    fn d45_predictor_4x4_libvpx_specialization() {
        // a=10 b=20 c=30 d=40 e=50 f=60 g=70 h=80
        let above = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let mut dst = [0u8; 16];
        d45_predictor_4x4(&mut dst, 4, &above);
        // AVG3(a,b,c)=20, AVG3(b,c,d)=30, AVG3(c,d,e)=40, AVG3(d,e,f)=50.
        // AVG3(e,f,g)=60, AVG3(f,g,h)=70.
        // DST(3, 3) = h = 80.
        let expected: [u8; 16] = [
            20, 30, 40, 50, // row 0 — col 3 = AVG3(d,e,f)=50 (generic = D = 40)
            30, 40, 50, 60, // row 1
            40, 50, 60, 70, // row 2
            50, 60, 70, 80, // row 3 — col 3 = h = 80
        ];
        assert_eq!(dst, expected);
    }

    /// Stage 3 stretch M.15 — D207 4x4 specialization.
    /// libvpx `vpx_d207_predictor_4x4_c` writes the bottom-right 2x2
    /// block as solid `L = left[3]` (incl. DST(2, 2) and DST(2, 3)),
    /// which matches the generic for left walks where `left[bs-1]`
    /// dominates — but the cell-by-cell write order is explicit.
    #[test]
    fn d207_predictor_4x4_libvpx_specialization() {
        // i=10 j=20 k=30 l=40
        let left = [10u8, 20, 30, 40];
        let mut dst = [0u8; 16];
        d207_predictor_4x4(&mut dst, 4, &left);
        // DST(0, 0)=AVG2(I, J)=15
        // DST(1, 0)=AVG3(I, J, K)=(10+40+30+2)>>2=20
        // DST(2, 0)=DST(0, 1)=AVG2(J, K)=25
        // DST(3, 0)=DST(1, 1)=AVG3(J, K, L)=(20+60+40+2)>>2=30
        // DST(2, 1)=DST(0, 2)=AVG2(K, L)=35
        // DST(3, 1)=DST(1, 2)=AVG3(K, L, L)=(30+80+40+2)>>2=38
        // DST(2, 2..3)=DST(3, 2..3)=DST(0..1, 3)=L=40
        let expected: [u8; 16] = [
            15, 20, 25, 30, // row 0
            25, 30, 35, 38, // row 1
            35, 38, 40, 40, // row 2
            40, 40, 40, 40, // row 3
        ];
        assert_eq!(dst, expected);
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
    fn d117_predictor_constant_input_is_constant() {
        // Constant neighbours (above_left = above = left = 100) should
        // yield a constant 100 block: every AVG2/AVG3 of 100s is 100.
        let above = [100u8; 8];
        let left = [100u8; 4];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 100,
        };
        let mut dst = [0u8; 16];
        d117_predictor(&mut dst, 4, 4, ctx);
        assert_eq!(dst, [100u8; 16]);
    }

    #[test]
    fn d117_predictor_4x4_hand_traced_row_0_and_col_0() {
        // above_left = 10, above = [20, 30, 40, 50], left = [60, 70, 80, 90].
        // Row 0: AVG2 of pairs starting from above_left.
        //   dst[0] = AVG2(10, 20) = (10+20+1)>>1 = 15
        //   dst[1] = AVG2(20, 30) = 25
        //   dst[2] = AVG2(30, 40) = 35
        //   dst[3] = AVG2(40, 50) = 45
        // Row 1 col 0: AVG3(left[0]=60, above_left=10, above[0]=20)
        //            = (60 + 20 + 20 + 2) >> 2 = 102 >> 2 = 25
        // Row 1 col 1: AVG3(above_left=10, above[0]=20, above[1]=30)
        //            = (10 + 40 + 30 + 2) >> 2 = 82 >> 2 = 20
        // Row 1 col 2: AVG3(above[0]=20, above[1]=30, above[2]=40)
        //            = (20 + 60 + 40 + 2) >> 2 = 122 >> 2 = 30
        // Row 1 col 3: AVG3(above[1]=30, above[2]=40, above[3]=50)
        //            = (30 + 80 + 50 + 2) >> 2 = 162 >> 2 = 40
        // Row 2 col 0: AVG3(above_left=10, left[0]=60, left[1]=70)
        //            = (10 + 120 + 70 + 2) >> 2 = 202 >> 2 = 50
        // Row 3 col 0: AVG3(left[0]=60, left[1]=70, left[2]=80)
        //            = (60 + 140 + 80 + 2) >> 2 = 282 >> 2 = 70
        let above = [20u8, 30, 40, 50];
        let left = [60u8, 70, 80, 90];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 10,
        };
        let mut dst = [0u8; 16];
        d117_predictor(&mut dst, 4, 4, ctx);
        let at = |r: usize, c: usize| dst[r * 4 + c];
        assert_eq!([at(0, 0), at(0, 1), at(0, 2), at(0, 3)], [15, 25, 35, 45]);
        assert_eq!(at(1, 0), 25);
        assert_eq!(at(1, 1), 20);
        assert_eq!(at(1, 2), 30);
        assert_eq!(at(1, 3), 40);
        assert_eq!(at(2, 0), 50);
        assert_eq!(at(3, 0), 70);
    }

    #[test]
    fn d135_predictor_constant_input_is_constant() {
        let above = [100u8; 4];
        let left = [100u8; 4];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 100,
        };
        let mut dst = [0u8; 16];
        d135_predictor(&mut dst, 4, 4, ctx);
        assert_eq!(dst, [100u8; 16]);
    }

    #[test]
    fn d135_predictor_4x4_diagonal_property() {
        // libvpx D135 is a true 135° diagonal. With distinctive
        // above_left = 50, above row = [60, 70, 80, 90], left col
        // = [40, 30, 20, 10], the diagonal cells dst[i][i] should
        // all carry roughly the smoothed above_left value
        // AVG3(left[0], above_left, above[0])
        //   = (40 + 100 + 60 + 2) >> 2 = 202 >> 2 = 50.
        let above = [60u8, 70, 80, 90];
        let left = [40u8, 30, 20, 10];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 50,
        };
        let mut dst = [0u8; 16];
        d135_predictor(&mut dst, 4, 4, ctx);
        let at = |r: usize, c: usize| dst[r * 4 + c];
        // Diagonal from (0,0) all carries border[bs-1] = AVG3 of
        // (left[0]=40, above_left=50, above[0]=60) = (40+100+60+2)>>2 = 50.
        assert_eq!(at(0, 0), 50);
        assert_eq!(at(1, 1), 50);
        assert_eq!(at(2, 2), 50);
        assert_eq!(at(3, 3), 50);
    }

    /// Stage 3 stretch M.18 — verify our generic [`d135_predictor`] at
    /// bs=4 produces the exact bytes that libvpx's hand-unrolled
    /// `vpx_d135_predictor_4x4_c` emits for the captured edges of
    /// vp90-2-05-resize frame 1 Block4x8 D135 at (px 276, row 0).
    ///
    /// libvpx wires `vpx_d135_predictor_4x4_c` (intrapred.c:411-430) into
    /// `pred[D135][TX_4X4]` via `intra_pred_no_4x4(d135)` — bs=4 D135 in
    /// libvpx **never** runs through the generic kernel.  M.15's claim
    /// that the spec "coincides with generic" was a hand-trace; M.18
    /// mechanically verifies it for these specific edges via the C
    /// oracle (see `tools/libvpx_decode_coefs_oracle/oracle.c` M.18 block).
    /// This unit test pins the rvp9 side of that mechanical equality.
    ///
    /// Sub-0 (rows 0..3 of the Block4x8):
    ///   above_left=127, above=[127,127,127,127], left=[73,73,73,73].
    /// Expected output (from libvpx oracle):
    ///   row 0: [114, 127, 127, 127]
    ///   row 1: [ 87, 114, 127, 127]
    ///   row 2: [ 73,  87, 114, 127]
    ///   row 3: [ 73,  73,  87, 114]
    #[test]
    fn d135_predictor_4x4_matches_libvpx_spec_m18_resize_sub0() {
        let above = [127u8, 127, 127, 127];
        let left = [73u8, 73, 73, 73];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 127,
        };
        let mut dst = [0u8; 16];
        d135_predictor(&mut dst, 4, 4, ctx);
        let expect: [u8; 16] = [
            114, 127, 127, 127, 87, 114, 127, 127, 73, 87, 114, 127, 73, 73, 87, 114,
        ];
        assert_eq!(
            dst, expect,
            "D135 4x4 must match libvpx vpx_d135_predictor_4x4_c"
        );
    }

    /// Stage 3 stretch M.18 — second sub-block (rows 4..7) of the same
    /// Block4x8 D135 at (px 276, row 0).  Edges:
    ///   above_left=73, above=[85,76,60,61], left=[49,49,49,49].
    /// Expected from libvpx oracle:
    ///   row 0: [70, 80, 74, 64]   row 1: [55, 70, 80, 74]
    ///   row 2: [49, 55, 70, 80]   row 3: [49, 49, 55, 70]
    #[test]
    fn d135_predictor_4x4_matches_libvpx_spec_m18_resize_sub1() {
        let above = [85u8, 76, 60, 61];
        let left = [49u8, 49, 49, 49];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 73,
        };
        let mut dst = [0u8; 16];
        d135_predictor(&mut dst, 4, 4, ctx);
        let expect: [u8; 16] = [
            70, 80, 74, 64, 55, 70, 80, 74, 49, 55, 70, 80, 49, 49, 55, 70,
        ];
        assert_eq!(
            dst, expect,
            "D135 4x4 must match libvpx vpx_d135_predictor_4x4_c"
        );
    }

    /// Stage 3 stretch M.18 — paranoia cross-check for D117 and D153
    /// 4×4 specs.  M.15 hand-claimed both "coincide with generic at
    /// bs=4"; M.18 picks arbitrary non-trivial edges (al=11,
    /// above=[37,53,71,89], left=[17,23,29,41]) and pins our generic
    /// kernels' output against libvpx's hand-unrolled spec values
    /// captured by the oracle.  If a future libvpx revision diverges,
    /// these tests trip immediately.
    #[test]
    fn d117_predictor_4x4_coincides_with_libvpx_spec() {
        let above = [37u8, 53, 71, 89];
        let left = [17u8, 23, 29, 41];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 11,
        };
        let mut dst = [0u8; 16];
        d117_predictor(&mut dst, 4, 4, ctx);
        // Values captured from vpx_d117_predictor_4x4_c via M.18 oracle.
        let expect: [u8; 16] = [
            24, 45, 62, 80, 19, 35, 54, 71, 17, 24, 45, 62, 23, 19, 35, 54,
        ];
        assert_eq!(
            dst, expect,
            "D117 generic at bs=4 must match libvpx 4x4 spec"
        );
    }

    #[test]
    fn d153_predictor_4x4_coincides_with_libvpx_spec() {
        let above = [37u8, 53, 71, 89];
        let left = [17u8, 23, 29, 41];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 11,
        };
        let mut dst = [0u8; 16];
        d153_predictor(&mut dst, 4, 4, ctx);
        // Values captured from vpx_d153_predictor_4x4_c via M.18 oracle.
        let expect: [u8; 16] = [
            14, 19, 35, 54, 20, 17, 14, 19, 26, 23, 20, 17, 35, 31, 26, 23,
        ];
        assert_eq!(
            dst, expect,
            "D153 generic at bs=4 must match libvpx 4x4 spec"
        );
    }

    #[test]
    fn d153_predictor_constant_input_is_constant() {
        let above = [100u8; 4];
        let left = [100u8; 4];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 100,
        };
        let mut dst = [0u8; 16];
        d153_predictor(&mut dst, 4, 4, ctx);
        assert_eq!(dst, [100u8; 16]);
    }

    #[test]
    fn d153_predictor_4x4_col_0_and_col_1_hand_traced() {
        // above_left = 10, above = [20, 30, 40, 50], left = [60, 70, 80, 90].
        // Col 0:
        //   dst[0][0] = AVG2(above_left=10, left[0]=60) = (10+60+1)>>1 = 35
        //   dst[1][0] = AVG2(left[0]=60, left[1]=70) = (60+70+1)>>1 = 65
        //   dst[2][0] = AVG2(left[1]=70, left[2]=80) = 75
        //   dst[3][0] = AVG2(left[2]=80, left[3]=90) = 85
        // Col 1:
        //   dst[0][1] = AVG3(left[0]=60, above_left=10, above[0]=20)
        //             = (60+20+20+2)>>2 = 102>>2 = 25
        //   dst[1][1] = AVG3(above_left=10, left[0]=60, left[1]=70)
        //             = (10+120+70+2)>>2 = 202>>2 = 50
        //   dst[2][1] = AVG3(left[0]=60, left[1]=70, left[2]=80)
        //             = (60+140+80+2)>>2 = 70
        //   dst[3][1] = AVG3(left[1]=70, left[2]=80, left[3]=90)
        //             = (70+160+90+2)>>2 = 80
        let above = [20u8, 30, 40, 50];
        let left = [60u8, 70, 80, 90];
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 10,
        };
        let mut dst = [0u8; 16];
        d153_predictor(&mut dst, 4, 4, ctx);
        let at = |r: usize, c: usize| dst[r * 4 + c];
        assert_eq!(at(0, 0), 35);
        assert_eq!(at(1, 0), 65);
        assert_eq!(at(2, 0), 75);
        assert_eq!(at(3, 0), 85);
        assert_eq!(at(0, 1), 25);
        assert_eq!(at(1, 1), 50);
        assert_eq!(at(2, 1), 70);
        assert_eq!(at(3, 1), 80);
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

            v_predictor(
                &mut dst,
                bs,
                bs,
                PredCtx {
                    above: &above,
                    left: &left,
                    above_left: 50,
                },
            );
            // Row 0 == above.
            assert_eq!(&dst[0..bs], &above[..]);

            h_predictor(
                &mut dst,
                bs,
                bs,
                PredCtx {
                    above: &above,
                    left: &left,
                    above_left: 50,
                },
            );
            // Column 0 == left.
            for r in 0..bs {
                assert_eq!(dst[r * bs], left[r]);
            }

            dc_predictor(
                &mut dst,
                bs,
                bs,
                PredCtx {
                    above: &above,
                    left: &left,
                    above_left: 50,
                },
            );
            // DC: rounded average of 2*bs values.
            let sum: u32 = above.iter().chain(left.iter()).map(|&v| v as u32).sum();
            let dc = ((sum + bs as u32) / (2 * bs as u32)) as u8;
            assert!(dst.iter().all(|&v| v == dc));
        }
    }

    // -------- Edge-builder (`build_intra_predictors`) --------

    use crate::transforms::TxSize;

    /// Synthesise a plane: stride 64, height 64, fill cell `(y, x)` with
    /// `(10 + y * 3 + x * 5) as u8` so every cell is uniquely identifiable.
    fn synth_plane() -> (Vec<u8>, usize, usize, usize) {
        let stride = 64;
        let width = 64;
        let height = 64;
        let mut buf = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                buf[y * stride + x] = (10u32 + y as u32 * 3 + x as u32 * 5) as u8;
            }
        }
        (buf, stride, width, height)
    }

    #[test]
    fn build_predictors_all_unavailable_yields_127_129() {
        // Top-left frame corner with neither neighbour available — DC mode
        // (NEED_ABOVE | NEED_LEFT). above and above_left should be 127;
        // left should be 129. (Note: when have_top=false AND have_left=false,
        // above_data[15] is set to 127 because the NEED_ABOVE branch sets it.)
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        // Use (1, 1) so the plane bounds-check still passes; but with
        // have_top = have_left = false the function must NOT read pixels.
        build_intra_predictors(
            &plane,
            stride,
            /*px_x=*/ 1,
            /*px_y=*/ 1,
            w,
            h,
            TxSize::Tx4x4,
            IntraMode::Dc,
            /*have_top=*/ false,
            /*have_left=*/ false,
            /*have_right=*/ false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(4);
        // DC mode (NEED_ABOVE | NEED_LEFT) only fills `bs` above samples;
        // the trailing `bs` slots in the 2*bs window are not touched by
        // the edge-builder. Compare only the bytes the kernel will read.
        assert_eq!(&ctx.above[..4], &[127u8; 4][..]);
        assert_eq!(ctx.left, &[129u8; 4][..]);
        assert_eq!(ctx.above_left, 127);
    }

    #[test]
    fn build_predictors_top_only_above_from_plane_left_is_129() {
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 4;
        let (px_x, px_y) = (10, 5);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx4x4,
            IntraMode::Dc,
            true,
            false,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        // above row = plane[(px_y - 1) * stride + px_x .. + bs]
        let expected_above: Vec<u8> = (0..bs)
            .map(|i| plane[(px_y - 1) * stride + px_x + i])
            .collect();
        assert_eq!(&ctx.above[..bs], &expected_above[..]);
        assert_eq!(ctx.left, &[129u8; 4][..]);
        // above_left: have_left=false → 129.
        assert_eq!(ctx.above_left, 129);
    }

    #[test]
    fn build_predictors_left_only_left_from_plane_above_is_127() {
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 4;
        let (px_x, px_y) = (8, 8);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx4x4,
            IntraMode::Dc,
            false,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        assert_eq!(&ctx.above[..bs], &[127u8; 4][..]);
        let expected_left: Vec<u8> = (0..bs)
            .map(|i| plane[(px_y + i) * stride + px_x - 1])
            .collect();
        assert_eq!(ctx.left, &expected_left[..]);
        // NEED_ABOVE branch fires → above_left = 127.
        assert_eq!(ctx.above_left, 127);
    }

    #[test]
    fn build_predictors_both_available_above_left_from_plane() {
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 4;
        let (px_x, px_y) = (8, 8);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx4x4,
            IntraMode::Dc,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let expected_above: Vec<u8> = (0..bs)
            .map(|i| plane[(px_y - 1) * stride + px_x + i])
            .collect();
        let expected_left: Vec<u8> = (0..bs)
            .map(|i| plane[(px_y + i) * stride + px_x - 1])
            .collect();
        assert_eq!(&ctx.above[..bs], &expected_above[..]);
        assert_eq!(ctx.left, &expected_left[..]);
        // above_left = plane[(px_y - 1) * stride + (px_x - 1)].
        assert_eq!(ctx.above_left, plane[(px_y - 1) * stride + (px_x - 1)]);
    }

    #[test]
    fn build_predictors_d45_above_right_replicated_without_have_right() {
        // D45 needs 2*bs above pixels; with bs=8 the bs==4 fast-path
        // does not apply, so above[bs..2*bs] are replicated from
        // above[bs-1] regardless of `have_right`.
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 8;
        let (px_x, px_y) = (16, 8);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx8x8,
            IntraMode::D45,
            true,
            true,
            /*have_right=*/ false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let above_ref_start = (px_y - 1) * stride + px_x;
        // Above row [0..bs] is the in-frame row.
        for i in 0..bs {
            assert_eq!(ctx.above[i], plane[above_ref_start + i]);
        }
        // Above row [bs..2*bs] is the replicated last byte.
        let last = plane[above_ref_start + bs - 1];
        for i in bs..2 * bs {
            assert_eq!(ctx.above[i], last, "above[{i}] should be replicated");
        }
        assert_eq!(ctx.above_left, plane[above_ref_start - 1]);
    }

    #[test]
    fn build_predictors_d45_4x4_have_right_reads_above_right_pixels() {
        // bs == 4 + have_right + not right_clipped → libvpx fast path
        // copies the next bs pixels from the plane (vp9_reconintra.c:383-384).
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 4;
        let (px_x, px_y) = (8, 4);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx4x4,
            IntraMode::D45,
            true,
            true,
            /*have_right=*/ true,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let above_ref_start = (px_y - 1) * stride + px_x;
        for i in 0..2 * bs {
            assert_eq!(
                ctx.above[i],
                plane[above_ref_start + i],
                "above[{i}] should come from plane (have_right fast path)"
            );
        }
    }

    #[test]
    fn build_predictors_left_unavailable_above_left_falls_to_129() {
        // V_PRED only sets above/above_left (NEED_ABOVE, no NEED_LEFT).
        // With have_left=false, above_left should be 129.
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 8;
        let (px_x, px_y) = (0, 8);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx8x8,
            IntraMode::V,
            /*have_top=*/ true,
            /*have_left=*/ false,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        assert_eq!(ctx.above_left, 129);
        // The above row comes from the plane.
        let above_ref_start = (px_y - 1) * stride + px_x;
        for i in 0..bs {
            assert_eq!(ctx.above[i], plane[above_ref_start + i]);
        }
    }

    #[test]
    fn build_predictors_right_clipped_replicates_last_in_frame_sample() {
        // px_x + bs > plane_width forces the slow path. We use a
        // narrower synthetic plane.
        let stride = 32;
        let width = 12;
        let height = 32;
        let mut plane = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * stride + x] = (y as u32 * 7 + x as u32 * 11 + 3) as u8;
            }
        }
        let mut edges = EdgeBuffers::new();
        let bs = 8;
        // Block at (px_x=8, px_y=8): px_x + bs = 16 > width=12 → slow path.
        let (px_x, px_y) = (8, 8);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            width,
            height,
            TxSize::Tx8x8,
            IntraMode::V,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let above_ref_start = (px_y - 1) * stride + px_x;
        let r = width - px_x; // 4 in-frame samples
        for i in 0..r {
            assert_eq!(ctx.above[i], plane[above_ref_start + i]);
        }
        let last = plane[above_ref_start + r - 1];
        for i in r..bs {
            assert_eq!(ctx.above[i], last, "above[{i}] replicated");
        }
    }

    #[test]
    fn build_predictors_bottom_clipped_replicates_last_left_sample() {
        // px_y + bs > plane_height forces the slow left-column path.
        let stride = 32;
        let width = 32;
        let height = 12;
        let mut plane = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * stride + x] = (y as u32 * 13 + x as u32 * 17 + 1) as u8;
            }
        }
        let mut edges = EdgeBuffers::new();
        let bs = 8;
        // Block at (px_x=4, px_y=8): px_y + bs = 16 > height=12 → slow.
        let (px_x, px_y) = (4, 8);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            width,
            height,
            TxSize::Tx8x8,
            IntraMode::H,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let extend_bottom = height - px_y; // 4 in-frame rows
        for i in 0..extend_bottom {
            let want = plane[(px_y + i) * stride + (px_x - 1)];
            assert_eq!(ctx.left[i], want, "left[{i}] in-frame");
        }
        let last = plane[(px_y + extend_bottom - 1) * stride + (px_x - 1)];
        for i in extend_bottom..bs {
            assert_eq!(ctx.left[i], last, "left[{i}] replicated");
        }
    }

    #[test]
    fn build_predictors_tx16x16_full_in_frame() {
        // Larger tx_size — 16x16, fully in frame, both neighbours.
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 16;
        let (px_x, px_y) = (16, 16);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx16x16,
            IntraMode::Tm,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let above_ref_start = (px_y - 1) * stride + px_x;
        for i in 0..bs {
            assert_eq!(ctx.above[i], plane[above_ref_start + i], "above[{i}]");
        }
        for i in 0..bs {
            assert_eq!(
                ctx.left[i],
                plane[(px_y + i) * stride + px_x - 1],
                "left[{i}]"
            );
        }
        assert_eq!(ctx.above_left, plane[above_ref_start - 1]);
    }

    // ------------------------------------------------------------------
    // Stage 9 — dispatch-table sweep coverage
    //
    // Below: exhaustive sweeps over every `(IntraMode, BlockSize)` pair
    // exercising `intra_predict`'s dispatch table. Each test asserts
    // bit-identical output between the dispatched call and a direct
    // call to the scalar kernel for that mode at that size. This pins
    // every `match` arm in `intra_predict` so future SIMD wiring changes
    // can't silently mis-route a (mode, size) pair to the wrong kernel.
    // ------------------------------------------------------------------

    /// Build a synthetic edge context for a `bs × bs` block where
    /// `above`/`left`/`above_left` are uniquely identifiable. The
    /// `above` slice is sized for the directional `2*bs` consumers.
    fn make_edges_for_dispatch(bs: usize) -> (Vec<u8>, Vec<u8>, u8) {
        let above: Vec<u8> = (0..2 * bs as u8).map(|i| 10 + 3 * i).collect();
        let left: Vec<u8> = (0..bs as u8).map(|i| 50 + 5 * i).collect();
        let above_left = 7u8;
        (above, left, above_left)
    }

    /// Dispatch sweep: every (`IntraMode`, `bs`) pair through
    /// `intra_predict` must produce the same bytes as a direct kernel
    /// call. `bs == 4` D45/D63/D207 route through the `*_4x4`
    /// specialisations; `bs > 4` modes route through the generic kernels.
    #[test]
    fn intra_predict_dispatch_sweep_all_modes_all_sizes() {
        for &bs in &[4usize, 8, 16, 32] {
            let (above, left, al) = make_edges_for_dispatch(bs);
            let ctx = PredCtx {
                above: &above,
                left: &left,
                above_left: al,
            };
            for raw_mode in 0u8..=9 {
                let mode = IntraMode::from_u8(raw_mode).unwrap();

                let mut dst_disp = vec![0u8; bs * bs];
                intra_predict(
                    mode,
                    &mut dst_disp,
                    bs,
                    bs,
                    ctx,
                    DcAvailability {
                        above_available: true,
                        left_available: true,
                    },
                    &scalar_dispatch(),
                );

                // Hand-pick the scalar kernel matching the (mode, bs)
                // dispatch path inside `intra_predict`.
                let mut dst_direct = vec![0u8; bs * bs];
                match mode {
                    IntraMode::Dc => {
                        // bs=4 + both-avail routes through dc_predictor_4x4
                        // slot; all other bs go through the
                        // `dc_predict_with_availability` path. Both produce
                        // the same bytes as `dc_predictor`.
                        dc_predictor(&mut dst_direct, bs, bs, ctx);
                    }
                    IntraMode::V => v_predictor(&mut dst_direct, bs, bs, ctx),
                    IntraMode::H => h_predictor(&mut dst_direct, bs, bs, ctx),
                    IntraMode::Tm => tm_predictor(&mut dst_direct, bs, bs, ctx),
                    IntraMode::D45 => {
                        if bs == 4 {
                            d45_predictor_4x4(&mut dst_direct, bs, &above);
                        } else {
                            d45_predictor(&mut dst_direct, bs, bs, &above);
                        }
                    }
                    IntraMode::D63 => {
                        if bs == 4 {
                            d63_predictor_4x4(&mut dst_direct, bs, &above);
                        } else {
                            d63_predictor(&mut dst_direct, bs, bs, &above);
                        }
                    }
                    IntraMode::D207 => {
                        if bs == 4 {
                            d207_predictor_4x4(&mut dst_direct, bs, &left);
                        } else {
                            d207_predictor(&mut dst_direct, bs, bs, &left);
                        }
                    }
                    IntraMode::D135 => d135_predictor(&mut dst_direct, bs, bs, ctx),
                    IntraMode::D117 => d117_predictor(&mut dst_direct, bs, bs, ctx),
                    IntraMode::D153 => d153_predictor(&mut dst_direct, bs, bs, ctx),
                }

                assert_eq!(
                    dst_disp, dst_direct,
                    "dispatch mismatch at mode={mode:?}, bs={bs}",
                );
            }
        }
    }

    /// DC-availability dispatch sweep at every block size. Each of the
    /// four `(above_available, left_available)` arms routes to the
    /// correct kernel for every `bs ∈ {4, 8, 16, 32}`. With uniform
    /// above (=16) and left (=32) the DC values are hand-computable:
    /// both → 24; above-only → 16; left-only → 32; neither → 128.
    #[test]
    fn dc_predict_with_availability_sweep_all_sizes() {
        let dispatch = scalar_dispatch();
        for &bs in &[4usize, 8, 16, 32] {
            let above = vec![16u8; bs];
            let left = vec![32u8; bs];

            // Both available.
            let mut dst = vec![0u8; bs * bs];
            dc_predict_with_availability(
                &mut dst,
                bs,
                bs,
                &above,
                &left,
                DcAvailability {
                    above_available: true,
                    left_available: true,
                },
                &dispatch,
            );
            assert!(
                dst.iter().all(|&v| v == 24),
                "bs={bs} both-avail expected 24",
            );

            // Above only.
            let mut dst = vec![0u8; bs * bs];
            dc_predict_with_availability(
                &mut dst,
                bs,
                bs,
                &above,
                &left,
                DcAvailability {
                    above_available: true,
                    left_available: false,
                },
                &dispatch,
            );
            assert!(
                dst.iter().all(|&v| v == 16),
                "bs={bs} above-only expected 16",
            );

            // Left only.
            let mut dst = vec![0u8; bs * bs];
            dc_predict_with_availability(
                &mut dst,
                bs,
                bs,
                &above,
                &left,
                DcAvailability {
                    above_available: false,
                    left_available: true,
                },
                &dispatch,
            );
            assert!(
                dst.iter().all(|&v| v == 32),
                "bs={bs} left-only expected 32",
            );

            // Neither.
            let mut dst = vec![0u8; bs * bs];
            dc_predict_with_availability(
                &mut dst,
                bs,
                bs,
                &above,
                &left,
                DcAvailability {
                    above_available: false,
                    left_available: false,
                },
                &dispatch,
            );
            assert!(
                dst.iter().all(|&v| v == 128),
                "bs={bs} neither expected 128",
            );
        }
    }

    /// `extend_mask` covers every `IntraMode` arm — exercise each via
    /// `build_intra_predictors` with a unique mode-specific edge fill
    /// expectation. The point is to walk the `match mode` table in
    /// `extend_mask` for every variant.
    #[test]
    fn build_intra_predictors_extend_mask_sweep_all_modes_unavailable_neighbours() {
        // All-unavailable neighbours: every mode falls through the
        // `127`/`129` fill paths. Use a narrow plane that satisfies
        // the in-frame bound checks; have_top=false / have_left=false
        // forces the fallback path inside the NEED_* branches.
        let (plane, stride, w, h) = synth_plane();
        for raw_mode in 0u8..=9 {
            let mode = IntraMode::from_u8(raw_mode).unwrap();
            let mut edges = EdgeBuffers::new();
            build_intra_predictors(
                &plane,
                stride,
                1,
                1,
                w,
                h,
                TxSize::Tx4x4,
                mode,
                false,
                false,
                false,
                &mut edges,
            );
            let ctx = edges.pred_ctx(4);
            // Every mode's `above_left` resolves through ABOVE_UNAVAILABLE_FILL
            // (127) when no NEED_LEFT/NEED_ABOVE fallback wrote it — the
            // NEED_ABOVERIGHT mode (D45/D63) also writes 127 in the else
            // branch of its `have_top` guard.
            match mode {
                IntraMode::D207 => {
                    // NEED_LEFT only → left=129, above untouched.
                    assert_eq!(ctx.left, &[129u8; 4][..], "D207 left");
                }
                IntraMode::H => {
                    // NEED_LEFT only.
                    assert_eq!(ctx.left, &[129u8; 4][..], "H left");
                }
                IntraMode::V => {
                    // NEED_ABOVE only.
                    assert_eq!(&ctx.above[..4], &[127u8; 4][..], "V above");
                    assert_eq!(ctx.above_left, 127);
                }
                IntraMode::D45 | IntraMode::D63 => {
                    // NEED_ABOVERIGHT only → above [0..2*bs] = 127.
                    assert_eq!(&ctx.above[..8], &[127u8; 8][..], "{mode:?} above 2*bs");
                    assert_eq!(ctx.above_left, 127);
                }
                _ => {
                    // DC, D135, D117, D153, TM: NEED_LEFT | NEED_ABOVE.
                    assert_eq!(&ctx.above[..4], &[127u8; 4][..], "{mode:?} above");
                    assert_eq!(ctx.left, &[129u8; 4][..], "{mode:?} left");
                    assert_eq!(ctx.above_left, 127, "{mode:?} above_left");
                }
            }
        }
    }

    /// `build_intra_predictors` NEED_ABOVERIGHT fast-path with
    /// `bs == 4 + have_right + !right_clipped`: the `(bs == 4 &&
    /// have_right)` branch copies the next `bs` above-right pixels
    /// from the plane in one go.
    #[test]
    fn build_intra_predictors_d45_4x4_have_right_no_clip() {
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 4;
        // px_x + bs == 12 well within width=64; have_right=true →
        // fast-path that reads above[bs..2*bs] from the plane.
        let (px_x, px_y) = (8, 4);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx4x4,
            IntraMode::D45,
            true,
            true,
            true,
            &mut edges,
        );
        let above_ref_start = (px_y - 1) * stride + px_x;
        let ctx = edges.pred_ctx(bs);
        for i in 0..2 * bs {
            assert_eq!(
                ctx.above[i],
                plane[above_ref_start + i],
                "above[{i}] should be plane byte (fast path)",
            );
        }
    }

    /// `build_intra_predictors` NEED_ABOVERIGHT slow-path: the
    /// `right_clipped` (= `px_x + bs > plane_width`) branch with `bs == 4`.
    /// Hits the "fewer than bs pixels in-frame" sub-case where
    /// `px_x + bs > plane_width` makes the first two sub-cases
    /// (`px_x + 2*bs <= plane_width` and `px_x + bs <= plane_width`)
    /// both fail.
    #[test]
    fn build_intra_predictors_d45_4x4_fewer_than_bs_pixels_in_frame() {
        let stride = 32;
        let width = 6; // px_x=4, bs=4 → px_x + bs = 8 > 6
        let height = 32;
        let mut plane = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * stride + x] = (y * 11 + x * 13 + 3) as u8;
            }
        }
        let mut edges = EdgeBuffers::new();
        let bs = 4;
        let (px_x, px_y) = (4, 8);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            width,
            height,
            TxSize::Tx4x4,
            IntraMode::D45,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let r = width - px_x; // 2 in-frame samples
        let above_ref_start = (px_y - 1) * stride + px_x;
        for i in 0..r {
            assert_eq!(ctx.above[i], plane[above_ref_start + i]);
        }
        let last = plane[above_ref_start + r - 1];
        for i in r..2 * bs {
            assert_eq!(ctx.above[i], last, "above[{i}] replicated");
        }
    }

    /// `build_intra_predictors` NEED_ABOVERIGHT slow path at bs=8:
    /// `right_clipped` triggers, falls through to the "fewer than bs
    /// pixels in-frame" branch (the only slow-path branch reachable
    /// given the `right_clipped = px_x + bs > plane_width` precondition).
    #[test]
    fn build_intra_predictors_d45_8x8_fewer_than_bs_in_frame_slow_path() {
        let stride = 32;
        let width = 12; // px_x=8, bs=8 → px_x + bs = 16 > 12
        let height = 32;
        let mut plane = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * stride + x] = (y * 7 + x * 11 + 2) as u8;
            }
        }
        let mut edges = EdgeBuffers::new();
        let bs = 8;
        let (px_x, px_y) = (8, 8);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            width,
            height,
            TxSize::Tx8x8,
            IntraMode::D45,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let above_ref_start = (px_y - 1) * stride + px_x;
        let r = width - px_x; // 4
        for i in 0..r {
            assert_eq!(ctx.above[i], plane[above_ref_start + i]);
        }
        let last = plane[above_ref_start + r - 1];
        for i in r..2 * bs {
            assert_eq!(ctx.above[i], last, "above[{i}] replicated");
        }
    }

    /// `build_intra_predictors` NEED_ABOVERIGHT fast-path **without**
    /// `have_right`: exercises the `else` branch of the
    /// `bs == 4 && have_right` ternary that replicates the trailing
    /// `bs` pixels from the last in-frame above byte.
    #[test]
    fn build_intra_predictors_d45_fast_path_no_have_right_replicates() {
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 4;
        let (px_x, px_y) = (8, 4);
        // have_right=false → trailing bs bytes replicate from above[bs-1].
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx4x4,
            IntraMode::D45,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let above_ref_start = (px_y - 1) * stride + px_x;
        for i in 0..bs {
            assert_eq!(ctx.above[i], plane[above_ref_start + i]);
        }
        let last = plane[above_ref_start + bs - 1];
        for i in bs..2 * bs {
            assert_eq!(ctx.above[i], last);
        }
    }

    /// `build_intra_predictors` NEED_ABOVERIGHT fewer-than-bs path
    /// where `have_right=false` (the `have_right=true` variant of this
    /// path doesn't exist — the fast-paths are only taken when
    /// `have_right == true` AND `bs == 4`, and `bs == 4` + fewer-than-bs
    /// in-frame is what the no_have_right path already covers).
    /// This test pins the `bs > 4` slow-path's fewer-than-bs branch.
    #[test]
    fn build_intra_predictors_d45_8x8_fewer_than_bs_pixels_in_frame() {
        let stride = 32;
        let width = 12; // px_x=8, bs=8 → px_x + bs = 16 > 12, r=4 < bs
        let height = 32;
        let mut plane = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * stride + x] = (y * 7 + x * 11 + 2) as u8;
            }
        }
        let mut edges = EdgeBuffers::new();
        let bs = 8;
        let (px_x, px_y) = (8, 8);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            width,
            height,
            TxSize::Tx8x8,
            IntraMode::D45,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let above_ref_start = (px_y - 1) * stride + px_x;
        let r = width - px_x; // 4
        for i in 0..r {
            assert_eq!(ctx.above[i], plane[above_ref_start + i]);
        }
        let last = plane[above_ref_start + r - 1];
        for i in r..2 * bs {
            assert_eq!(ctx.above[i], last, "above[{i}] replicated");
        }
    }

    /// `build_intra_predictors` NEED_ABOVE right-clipped slow path with
    /// `have_top == true`: `px_x + bs > plane_width` triggers the
    /// in-frame replication branch on the NEED_ABOVE path (separate
    /// from the NEED_ABOVERIGHT path covered above).
    #[test]
    fn build_intra_predictors_v_right_clipped_with_have_left() {
        let stride = 32;
        let width = 12;
        let height = 32;
        let mut plane = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * stride + x] = (y * 5 + x * 7 + 3) as u8;
            }
        }
        let mut edges = EdgeBuffers::new();
        let bs = 8;
        let (px_x, px_y) = (8, 4);
        // V (NEED_ABOVE), have_left=true → above_left comes from plane.
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            width,
            height,
            TxSize::Tx8x8,
            IntraMode::V,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let above_ref_start = (px_y - 1) * stride + px_x;
        let r = width - px_x; // 4
        for i in 0..r {
            assert_eq!(ctx.above[i], plane[above_ref_start + i]);
        }
        let last = plane[above_ref_start + r - 1];
        for i in r..bs {
            assert_eq!(ctx.above[i], last);
        }
        // above_left from plane (have_left=true).
        assert_eq!(ctx.above_left, plane[above_ref_start - 1]);
    }

    /// `build_intra_predictors` H mode left fallback when `have_left=false`:
    /// every left[i] becomes LEFT_UNAVAILABLE_FILL = 129.
    #[test]
    fn build_intra_predictors_h_have_left_false() {
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        build_intra_predictors(
            &plane,
            stride,
            8,
            8,
            w,
            h,
            TxSize::Tx8x8,
            IntraMode::H,
            true,
            false,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(8);
        assert_eq!(ctx.left, &[129u8; 8][..]);
    }

    /// `build_intra_predictors` NEED_ABOVERIGHT (D45) with
    /// `have_left=false`: `above_left` falls to
    /// `LEFT_UNAVAILABLE_FILL = 129`. Pins the `have_left == false`
    /// arm of the NEED_ABOVERIGHT `above_row[-1]` write.
    #[test]
    fn build_intra_predictors_d45_have_left_false_uses_129() {
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 4;
        let (px_x, px_y) = (8, 4);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx4x4,
            IntraMode::D45,
            true,
            /*have_left=*/ false,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        // above row comes from the plane; above_left is 129 because
        // have_left=false.
        assert_eq!(ctx.above_left, 129);
    }

    /// `build_intra_predictors` D207 (NEED_LEFT only) with bottom-clipped
    /// plane — exercises the "extend_bottom < bs" left-column slow path
    /// in the NEED_LEFT branch.
    #[test]
    fn build_intra_predictors_d207_bottom_clipped() {
        let stride = 32;
        let width = 32;
        let height = 12;
        let mut plane = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * stride + x] = (y * 2 + x * 3 + 4) as u8;
            }
        }
        let mut edges = EdgeBuffers::new();
        let bs = 8;
        let (px_x, px_y) = (4, 8); // px_y + bs = 16 > 12 → slow path
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            width,
            height,
            TxSize::Tx8x8,
            IntraMode::D207,
            true,
            true,
            false,
            &mut edges,
        );
        let ctx = edges.pred_ctx(bs);
        let extend_bottom = height - px_y;
        for i in 0..extend_bottom {
            assert_eq!(ctx.left[i], plane[(px_y + i) * stride + (px_x - 1)]);
        }
        let last = plane[(px_y + extend_bottom - 1) * stride + (px_x - 1)];
        for i in extend_bottom..bs {
            assert_eq!(ctx.left[i], last);
        }
    }

    #[test]
    fn build_predictors_then_intra_predict_round_trip() {
        // End-to-end: build_intra_predictors → intra_predict produces
        // the same V_PRED output as calling intra_predict with a
        // hand-constructed PredCtx.
        let (plane, stride, w, h) = synth_plane();
        let mut edges = EdgeBuffers::new();
        let bs = 4;
        let (px_x, px_y) = (4, 4);
        build_intra_predictors(
            &plane,
            stride,
            px_x,
            px_y,
            w,
            h,
            TxSize::Tx4x4,
            IntraMode::V,
            true,
            true,
            false,
            &mut edges,
        );
        let mut dst_from_builder = [0u8; 16];
        let avail = DcAvailability {
            above_available: true,
            left_available: true,
        };
        let dispatch = scalar_dispatch();
        intra_predict(
            IntraMode::V,
            &mut dst_from_builder,
            4,
            bs,
            edges.pred_ctx(bs),
            avail,
            &dispatch,
        );

        // Reference path: build PredCtx by hand.
        let above_ref_start = (px_y - 1) * stride + px_x;
        let above: Vec<u8> = (0..bs).map(|i| plane[above_ref_start + i]).collect();
        let left: Vec<u8> = (0..bs)
            .map(|i| plane[(px_y + i) * stride + px_x - 1])
            .collect();
        let manual_ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: plane[above_ref_start - 1],
        };
        let mut dst_manual = [0u8; 16];
        intra_predict(
            IntraMode::V,
            &mut dst_manual,
            4,
            bs,
            manual_ctx,
            avail,
            &dispatch,
        );
        assert_eq!(dst_from_builder, dst_manual);
    }
}
