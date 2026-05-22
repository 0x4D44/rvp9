//! SB-level integer-projection motion estimation
//! (`vp9_int_pro_motion_estimation`).
//!
//! Port of `vp9_int_pro_motion_estimation` (`vp9_mcomp.c:2264-2400`)
//! and its supporting helpers `vector_match`, `vpx_int_pro_row_c`,
//! `vpx_int_pro_col_c`, `vpx_vector_var_c`
//! (`vp9_mcomp.c:2192-2254`, `vpx_dsp/avg.c:345-387`).
//!
//! The algorithm produces a cheap SB-level integer-pel MV by
//! comparing 1D row/column projections of source and reference
//! windows, rather than running a full 2D SAD search. The result
//! seeds the per-leaf picker's NEWMV diamond-search center via
//! `x->pred_mv[LAST_FRAME]` in libvpx.
//!
//! ## Why this matters
//!
//! Without an SB-level coarse-ME seed, the per-leaf NEWMV diamond
//! search starts from `(0, 0)` (or `nearest`, which is also `(0, 0)`
//! at frame boundaries / startup). At low Q, the MV-bit-cost of a
//! sub-pel NEWMV exceeds the SSE gain over NEARESTMV-at-(0, 0), so
//! the picker never selects NEWMV. With no NEWMV picks, the spatial
//! MV-ref propagation chain never fires either (every neighbour is
//! also NEARESTMV-at-(0, 0)). On continuous-motion content
//! (foreman_cif) this collapses inter prediction to "always copy LAST
//! frame" → PSNR decays linearly every frame.
//!
//! The libvpx fix: do a cheap 1D projection search at the 64×64 SB
//! level, get a starter MV, hand that to the per-leaf picker as the
//! NEWMV diamond seed. The seed has to actually beat NEARESTMV in
//! RD, which it can if the SB-level motion is large enough that a
//! single MV explains most of the SB.
//!
//! ## Algorithm
//!
//! For a 64×64 SB at `(sb_x, sb_y)`:
//!
//! 1. Build `hbuf[search_width]` = column-sums of the reference,
//!    starting at column `sb_x - bw/2`. Length is `search_width =
//!    2*bw`. Each entry sums `bh` rows. Each entry is normalised by
//!    `bh >> 1` (divisor matching libvpx `vpx_int_pro_row_c`).
//! 2. Build `vbuf[search_height]` = row-sums of the reference,
//!    starting at row `sb_y - bh/2`. Length is `search_height =
//!    2*bh`. Each entry sums `bw` columns. Each entry is normalised
//!    by `3 + (bw >> 5)` (libvpx's `norm_factor`).
//! 3. Build `src_hbuf[bw]` (column sums of source SB) and
//!    `src_vbuf[bh]` (row sums of source SB) with the same
//!    normalisations.
//! 4. `vector_match(hbuf, src_hbuf, bwl)` returns the best column
//!    shift in `[-bw/2, bw/2]` (full-pel units). Same for rows.
//! 5. Refine the (row, col) candidate with a 4-point diamond
//!    (`{-1, 0}`, `{0, -1}`, `{0, 1}`, `{1, 0}`) using the 2D SAD
//!    kernel for the SB size.
//! 6. Refine with a 2-point cross derived from the diamond results.
//! 7. Multiply by 8 to convert full-pel → Q3 (1/8-pel) units.
//! 8. Clamp to the subpel MV search range (caller's responsibility
//!    via [`MvLimits`]).
//!
//! ## Limitations / divergences
//!
//! - **8-bit only.** libvpx falls back to a single full-SB SAD for
//!   HBD; we don't ship that fallback because rvp9 only supports
//!   Profile 0 (8-bit).
//! - **No scaled-ref handling.** rvp9's single-LASTFRAME setup
//!   doesn't use scale factors; if/when GOLDEN / ALTREF land with
//!   resolution-scaled refs, this needs the
//!   `vp9_get_scaled_ref_frame` swap-out logic.

use crate::block::BlockSize;
use crate::dsp::distortion::{
    vpx_sad8x8, vpx_sad16x16, vpx_sad32x32, vpx_sad64x64,
};
use crate::mode_info::Mv;
use crate::mv_search::full_pixel::PlaneRef;

/// Output of [`int_pro_motion_estimation`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IntProMeResult {
    /// Resulting integer-pel MV (in Q3 / 1/8-pel units — i.e. the
    /// full-pel offset times 8). `Mv::ZERO` when the search was a
    /// no-op (e.g. SB clipped to frame edge) — caller treats
    /// `Mv::ZERO` as "no useful seed".
    pub mv: Mv,
    /// Best SAD over the SB at the returned MV. `u32::MAX` when the
    /// search was skipped (frame-edge clip).
    pub best_sad: u32,
}

impl IntProMeResult {
    /// Zero MV / no-search-performed sentinel.
    pub const ZERO: Self = Self {
        mv: Mv::ZERO,
        best_sad: u32::MAX,
    };
}

/// `vpx_int_pro_row_c` (`vpx_dsp/avg.c:345-359`). Sum the next
/// `16` columns of `ref` along `height` rows, normalise by
/// `height >> 1`, write into `hbuf[0..16]`.
///
/// Caller must ensure `ref[0 .. 16 + (height-1) * ref_stride]` is
/// in bounds.
#[inline]
fn vpx_int_pro_row(hbuf: &mut [i16], ref_plane: &[u8], ref_stride: usize, height: usize) {
    debug_assert_eq!(hbuf.len(), 16);
    debug_assert!(height >= 2);
    let norm_factor = (height >> 1) as i32;
    for idx in 0..16 {
        let mut sum: i32 = 0;
        for i in 0..height {
            sum += i32::from(ref_plane[i * ref_stride + idx]);
        }
        hbuf[idx] = (sum / norm_factor) as i16;
    }
}

/// `vpx_int_pro_col_c` (`vpx_dsp/avg.c:362-368`). Return the sum
/// of the next `width` columns of `ref`. Caller divides by the
/// matching `norm_factor`.
#[inline]
fn vpx_int_pro_col(ref_row: &[u8], width: usize) -> i32 {
    debug_assert!(ref_row.len() >= width);
    let mut sum: i32 = 0;
    for idx in 0..width {
        sum += i32::from(ref_row[idx]);
    }
    sum
}

/// `vpx_vector_var_c` (`vpx_dsp/avg.c:373-387`). Compute the
/// "variance" between two `4 << bwl`-length int16 vectors:
///
/// ```text
/// width = 4 << bwl
/// for i in 0..width: diff = ref[i] - src[i]; mean += diff; sse += diff*diff
/// var = sse - ((mean * mean) >> (bwl + 2))
/// ```
///
/// `bwl` ranges over `{1, 2, 3, 4}` (16, 32, 64 vs the 16/32/64
/// SB widths we care about). The result fits in i32 with margin
/// (sse: 26 bits dynamic range per libvpx comment).
#[inline]
fn vpx_vector_var(ref_vec: &[i16], src_vec: &[i16], bwl: u32) -> i32 {
    let width = 4usize << bwl;
    debug_assert!(ref_vec.len() >= width);
    debug_assert!(src_vec.len() >= width);
    let mut sse: i32 = 0;
    let mut mean: i32 = 0;
    for i in 0..width {
        let diff = i32::from(ref_vec[i]) - i32::from(src_vec[i]);
        mean += diff;
        sse += diff * diff;
    }
    sse - ((mean * mean) >> (bwl + 2))
}

/// `vector_match` (`vp9_mcomp.c:2192-2255`). Find the best 1D
/// translation in `[0, bw]` of a length-`bw` window of `src`
/// against `ref`, then return `center - (bw >> 1)` (i.e. the shift
/// centred around the SB's natural location).
///
/// `bwl` is `b_width_log2_lookup[bsize]` (= log2(bw / 4)). The
/// `ref` slice must be `>= 2*bw` long (libvpx's `search_width =
/// 2*bw`).
fn vector_match(ref_vec: &[i16], src_vec: &[i16], bwl: u32) -> i32 {
    let bw = 4i32 << bwl;
    let bw_usz = bw as usize;
    debug_assert!(ref_vec.len() >= 2 * bw_usz);
    debug_assert!(src_vec.len() >= bw_usz);

    let mut best_sad = i32::MAX;
    let mut offset = 0i32;

    // Coarse step: 16.
    let mut d = 0i32;
    while d <= bw {
        let this_sad = vpx_vector_var(&ref_vec[d as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            offset = d;
        }
        d += 16;
    }
    let mut center = offset;

    // Step: ±8.
    for d in [-8i32, 8i32] {
        let this_pos = offset + d;
        if this_pos < 0 || this_pos > bw {
            continue;
        }
        let this_sad = vpx_vector_var(&ref_vec[this_pos as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            center = this_pos;
        }
    }
    offset = center;

    // Step: ±4.
    for d in [-4i32, 4i32] {
        let this_pos = offset + d;
        if this_pos < 0 || this_pos > bw {
            continue;
        }
        let this_sad = vpx_vector_var(&ref_vec[this_pos as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            center = this_pos;
        }
    }
    offset = center;

    // Step: ±2.
    for d in [-2i32, 2i32] {
        let this_pos = offset + d;
        if this_pos < 0 || this_pos > bw {
            continue;
        }
        let this_sad = vpx_vector_var(&ref_vec[this_pos as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            center = this_pos;
        }
    }
    offset = center;

    // Step: ±1.
    for d in [-1i32, 1i32] {
        let this_pos = offset + d;
        if this_pos < 0 || this_pos > bw {
            continue;
        }
        let this_sad = vpx_vector_var(&ref_vec[this_pos as usize..], src_vec, bwl);
        if this_sad < best_sad {
            best_sad = this_sad;
            center = this_pos;
        }
    }

    center - (bw >> 1)
}

/// Pick the matching 2D SAD kernel for an SB-supported block size.
fn sad_block(bsize: BlockSize, src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u32 {
    match bsize {
        BlockSize::Block8x8 => vpx_sad8x8(src, src_stride, ref_block, ref_stride),
        BlockSize::Block16x16 => vpx_sad16x16(src, src_stride, ref_block, ref_stride),
        BlockSize::Block32x32 => vpx_sad32x32(src, src_stride, ref_block, ref_stride),
        BlockSize::Block64x64 => vpx_sad64x64(src, src_stride, ref_block, ref_stride),
        _ => u32::MAX,
    }
}

/// `b_width_log2_lookup[bsize]` for the square SB block sizes we
/// support. Mirrors libvpx `vp9_common_data.c:15`.
#[inline]
fn b_log2(bsize: BlockSize) -> u32 {
    match bsize {
        BlockSize::Block8x8 => 1,
        BlockSize::Block16x16 => 2,
        BlockSize::Block32x32 => 3,
        BlockSize::Block64x64 => 4,
        _ => 0,
    }
}

/// Port of `vp9_int_pro_motion_estimation` (`vp9_mcomp.c:2264-2400`).
///
/// Computes an integer-pel MV (returned in Q3 / 1/8-pel units) for
/// the source SB at `(sb_px_x, sb_px_y)` against the reference plane.
/// `bsize` must be one of the square sizes
/// `{Block8x8, Block16x16, Block32x32, Block64x64}`; other sizes
/// return [`IntProMeResult::ZERO`].
///
/// The returned MV is **unclamped**. Callers that need to feed it
/// into a per-leaf picker with an `MvLimits` envelope should clamp
/// it themselves (libvpx calls `clamp_mv` against the subpel-MV
/// search range derived from `vp9_set_subpel_mv_search_range`).
///
/// The reference plane must be the SAME dimensions as the source
/// plane (no scaled-ref support); the caller is responsible for
/// performing the equivalent of libvpx's `vp9_setup_pre_planes`
/// upstream.
#[must_use]
pub fn int_pro_motion_estimation(
    src_plane: &PlaneRef<'_>,
    ref_plane: &PlaneRef<'_>,
    sb_px_x: usize,
    sb_px_y: usize,
    bsize: BlockSize,
) -> IntProMeResult {
    let bw = usize::from(bsize.width());
    let bh = usize::from(bsize.height());
    if bw != bh
        || !matches!(
            bsize,
            BlockSize::Block8x8
                | BlockSize::Block16x16
                | BlockSize::Block32x32
                | BlockSize::Block64x64
        )
    {
        return IntProMeResult::ZERO;
    }

    // The 1D-projection search reads
    //   ref[sb_y - bh/2 .. sb_y + bh + bh/2 - 1]
    //      [sb_x - bw/2 .. sb_x + bw + bw/2 - 1]
    // and
    //   src[sb_y .. sb_y + bh - 1][sb_x .. sb_x + bw - 1].
    // The post-search 2D-SAD refinement reads an additional ±1
    // ref-pixel halo around the chosen MV. If any of these go OOB
    // we bail out with the ZERO sentinel — the picker just won't
    // get a seed for that SB (callers fall through to nearest).
    if sb_px_x < bw / 2 || sb_px_y < bh / 2 {
        return IntProMeResult::ZERO;
    }
    if sb_px_x + bw + bw / 2 > ref_plane.width
        || sb_px_y + bh + bh / 2 > ref_plane.height
    {
        return IntProMeResult::ZERO;
    }

    let search_width = bw << 1;
    let search_height = bh << 1;
    debug_assert!(search_width <= 128);
    debug_assert!(search_height <= 128);

    let src_stride = src_plane.stride;
    let ref_stride = ref_plane.stride;
    let bwl = b_log2(bsize);

    // ref_buf base for the row scan: starts (bw>>1) columns LEFT
    // of the SB. Each call to `vpx_int_pro_row` consumes 16 cols
    // and advances by 16, building `search_width` int16 entries.
    let mut hbuf = [0i16; 128];
    {
        let ref_top_left = sb_px_y * ref_stride + sb_px_x - (bw >> 1);
        let mut idx = 0;
        while idx < search_width {
            let off = ref_top_left + idx;
            vpx_int_pro_row(
                &mut hbuf[idx..idx + 16],
                &ref_plane.data[off..],
                ref_stride,
                bh,
            );
            idx += 16;
        }
    }

    // vbuf: row-sums of ref, starting (bh>>1) rows ABOVE the SB.
    let mut vbuf = [0i16; 128];
    {
        let norm_factor = 3 + (bw >> 5);
        let ref_top_left_row = sb_px_y - (bh >> 1);
        for idx in 0..search_height {
            let row_off = (ref_top_left_row + idx) * ref_stride + sb_px_x;
            let sum = vpx_int_pro_col(&ref_plane.data[row_off..], bw);
            vbuf[idx] = (sum >> norm_factor) as i16;
        }
    }

    // src_hbuf: column-sums of src SB. libvpx loops idx 0..bw step
    // 16 (Block16x16 → 1 iter, Block64x64 → 4 iters).
    let mut src_hbuf = [0i16; 64];
    if bw >= 16 {
        let mut idx = 0;
        while idx < bw {
            let off = sb_px_y * src_stride + sb_px_x + idx;
            vpx_int_pro_row(
                &mut src_hbuf[idx..idx + 16],
                &src_plane.data[off..],
                src_stride,
                bh,
            );
            idx += 16;
        }
    } else {
        // Block8x8: libvpx doesn't use int_pro for 8x8 (variance
        // partitioner only descends to 16x16 for the SB seed
        // call), but we keep a defensive scalar fallback so the
        // helper is total. Compute the 8 column-sums manually.
        let norm_factor = (bh >> 1) as i32;
        for idx in 0..bw {
            let mut sum: i32 = 0;
            for i in 0..bh {
                sum += i32::from(
                    src_plane.data[(sb_px_y + i) * src_stride + sb_px_x + idx],
                );
            }
            src_hbuf[idx] = (sum / norm_factor) as i16;
        }
    }

    // src_vbuf: row-sums of src SB.
    let mut src_vbuf = [0i16; 64];
    {
        let norm_factor = 3 + (bw >> 5);
        for idx in 0..bh {
            let row_off = (sb_px_y + idx) * src_stride + sb_px_x;
            let sum = vpx_int_pro_col(&src_plane.data[row_off..], bw);
            src_vbuf[idx] = (sum >> norm_factor) as i16;
        }
    }

    // 1D matches → integer-pel MV offsets in `[-bw/2, bw/2]`.
    let mut col = vector_match(&hbuf[..search_width], &src_hbuf[..bw], bwl);
    let mut row = vector_match(&vbuf[..search_height], &src_vbuf[..bh], bwl);

    // 2D refinement: 4-pt diamond + 2-pt cross, both gated by
    // staying inside the ref plane (with the ±1 halo the diamond
    // needs).
    let src_off = sb_px_y * src_stride + sb_px_x;
    let src_slice = &src_plane.data[src_off..];

    let try_sad = |row: i32, col: i32| -> Option<u32> {
        let abs_x = sb_px_x as i32 + col;
        let abs_y = sb_px_y as i32 + row;
        if abs_x < 0 || abs_y < 0 {
            return None;
        }
        if abs_x as usize + bw > ref_plane.width
            || abs_y as usize + bh > ref_plane.height
        {
            return None;
        }
        let off = (abs_y as usize) * ref_stride + (abs_x as usize);
        Some(sad_block(
            bsize,
            src_slice,
            src_stride,
            &ref_plane.data[off..],
            ref_stride,
        ))
    };

    let mut best_sad = match try_sad(row, col) {
        Some(sad) => sad,
        None => {
            // Initial 1D estimate lands OOB. Match the spirit of
            // libvpx's `clamp_mv` (which would silently recentre)
            // by falling back to (0, 0).
            row = 0;
            col = 0;
            try_sad(0, 0).unwrap_or(u32::MAX)
        }
    };

    // 4-pt diamond: { (-1,0), (0,-1), (0,1), (1,0) } (libvpx's
    // `search_pos` order, vp9_mcomp.c:2257-2262).
    let diamond = [(-1i32, 0i32), (0i32, -1i32), (0i32, 1i32), (1i32, 0i32)];
    let mut diamond_sads = [u32::MAX; 4];
    for (i, &(dr, dc)) in diamond.iter().enumerate() {
        if let Some(sad) = try_sad(row + dr, col + dc) {
            diamond_sads[i] = sad;
        }
    }

    // libvpx walks the 4 diamond results in order and updates
    // `*tmp_mv` (the running best) on strict `<`. `this_mv` is the
    // PRE-diamond center used by the cross-pos refinement
    // (vp9_mcomp.c:2345 `this_mv = *tmp_mv` taken BEFORE the
    // sdx4df update). We mirror that exactly.
    let pre_diamond_row = row;
    let pre_diamond_col = col;
    let mut best_row = row;
    let mut best_col = col;
    for (i, &(dr, dc)) in diamond.iter().enumerate() {
        if diamond_sads[i] < best_sad {
            best_sad = diamond_sads[i];
            best_row = pre_diamond_row + dr;
            best_col = pre_diamond_col + dc;
        }
    }

    // 2-pt cross derived from diamond SADs (libvpx
    // `vp9_mcomp.c:2369-2377`):
    //   this_sad[0] = up; this_sad[3] = down.
    //   this_sad[1] = left; this_sad[2] = right.
    // If up < down, this_mv.row -= 1; else this_mv.row += 1.
    // If left < right, this_mv.col -= 1; else this_mv.col += 1.
    let cross_row = if diamond_sads[0] < diamond_sads[3] {
        pre_diamond_row - 1
    } else {
        pre_diamond_row + 1
    };
    let cross_col = if diamond_sads[1] < diamond_sads[2] {
        pre_diamond_col - 1
    } else {
        pre_diamond_col + 1
    };
    if let Some(cross_sad) = try_sad(cross_row, cross_col) {
        if cross_sad < best_sad {
            best_sad = cross_sad;
            best_row = cross_row;
            best_col = cross_col;
        }
    }

    // Convert full-pel → Q3 (1/8-pel).
    let mv_row_q3 = (best_row * 8)
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    let mv_col_q3 = (best_col * 8)
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;

    IntProMeResult {
        mv: Mv {
            row: mv_row_q3,
            col: mv_col_q3,
        },
        best_sad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero motion: identical src and ref → search returns
    /// `Mv::ZERO` (with constant content the projections are all
    /// identical so vector_match's first-wins returns the centre
    /// = bw/2, which maps to MV = 0).
    #[test]
    fn zero_motion_yields_zero_mv() {
        let w = 256;
        let h = 256;
        let buf = vec![128u8; w * h];
        let src = PlaneRef { data: &buf, stride: w, width: w, height: h };
        let r = PlaneRef { data: &buf, stride: w, width: w, height: h };
        let result = int_pro_motion_estimation(&src, &r, 64, 64, BlockSize::Block64x64);
        assert_eq!(result.mv, Mv::ZERO);
    }

    /// Synthetic translation: ref is a shifted copy of src. The
    /// 1D-projection MV should point back to the shift (within
    /// the projection's accuracy).
    #[test]
    fn integer_translation_is_recovered() {
        let w = 256;
        let h = 256;
        let mut src_buf = vec![0u8; w * h];
        // Diagonal stripes — has horizontal + vertical structure
        // so both row and column projections are non-trivial.
        for y in 0..h {
            for x in 0..w {
                src_buf[y * w + x] = (((x + y) % 32) * 4) as u8;
            }
        }
        // Reference: ref[y][x] = src[y][x - shift]. So src's
        // feature at column x matches ref's feature at x + shift
        // → MV col = +shift.
        let shift = 4i32;
        let mut ref_buf = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let sx = x as i32 - shift;
                if sx < 0 || sx as usize >= w {
                    ref_buf[y * w + x] = 0;
                } else {
                    ref_buf[y * w + x] = src_buf[y * w + sx as usize];
                }
            }
        }
        let src = PlaneRef { data: &src_buf, stride: w, width: w, height: h };
        let r = PlaneRef { data: &ref_buf, stride: w, width: w, height: h };
        let result = int_pro_motion_estimation(&src, &r, 96, 96, BlockSize::Block64x64);
        // Q3 → full-pel; tolerance ±2 full-pels for the 1D
        // projection approximation.
        let col_fp = i32::from(result.mv.col) / 8;
        assert!(
            (col_fp - shift).abs() <= 2,
            "expected col MV near {} (full-pel), got {} ({} Q3)",
            shift,
            col_fp,
            result.mv.col
        );
    }

    /// SB at frame edge with insufficient left/top padding → ZERO
    /// sentinel (no seed).
    #[test]
    fn frame_edge_returns_zero_sentinel() {
        let w = 256;
        let h = 256;
        let buf = vec![128u8; w * h];
        let src = PlaneRef { data: &buf, stride: w, width: w, height: h };
        let r = PlaneRef { data: &buf, stride: w, width: w, height: h };
        // Block64x64 at (0, 0): needs 32 pixels of left padding,
        // which doesn't exist.
        let result = int_pro_motion_estimation(&src, &r, 0, 0, BlockSize::Block64x64);
        assert_eq!(result, IntProMeResult::ZERO);
    }

    /// vector_match centres at bw/2 → returns 0 for constant input.
    #[test]
    fn vector_match_constant_input_zero() {
        let bwl: u32 = 2;
        let bw = 4usize << bwl;
        let ref_vec = vec![100i16; 2 * bw];
        let src_vec = vec![100i16; bw];
        let mv = vector_match(&ref_vec, &src_vec, bwl);
        assert_eq!(mv, 0);
    }
}
