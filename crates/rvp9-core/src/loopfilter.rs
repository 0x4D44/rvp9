//! VP9 loop-filter threshold computation.
//!
//! Mirrors libvpx `vp9/common/vp9_loopfilter.c:212-250`
//! (`update_sharpness`, `vp9_loop_filter_init`).
//!
//! ## Scope of this commit (Task 6.1)
//!
//! Lands the per-filter-level threshold table:
//!
//! - [`LoopFilterThresh`] — the three thresholds (`mblim`, `lim`,
//!   `hev_thr`) used by every edge-filter kernel call.
//! - [`compute_thresholds`] — formula for one filter level given the
//!   frame's sharpness level (`lvl` and `sharpness_level` both come
//!   from the uncompressed header's loop-filter sub-block).
//! - [`LoopFilterInfo`] — per-frame table covering all 64 filter
//!   levels (`0..=MAX_LOOP_FILTER`).
//!
//! ## Edge-filter kernels
//!
//! Task 6.2a adds the four 4-tap edge filters — [`lpf_horizontal_4`],
//! [`lpf_horizontal_4_dual`], [`lpf_vertical_4`], [`lpf_vertical_4_dual`].
//! These are the workhorses for the smallest VP9 edge (4-pixel
//! transverse extent, 8-pixel longitudinal extent) and are the basis
//! for every coarser filter — `filter4` is the inner loop body that
//! the 8- and 16-tap kernels invoke after their own mask-and-flatness
//! checks.
//!
//! ## Deferred
//!
//! - The 8-tap and 16-tap edge filters (`lpf_*_{8,16}`) — Task 6.2b/c.
//! - Edge-mask machinery (`build_masks` and the
//!   `vp9_get_loop_filter_mask` function family) — Task 6.3.
//! - Per-segment / per-ref / per-mode filter-level resolution
//!   (`vp9_loop_filter_frame_init`) — Task 6.4.

/// Maximum filter level. Mirrors libvpx `MAX_LOOP_FILTER`
/// (`vp9_loopfilter.h:24`).
pub const MAX_LOOP_FILTER: u8 = 63;

/// SIMD broadcast width. libvpx ships each threshold as a
/// 16-byte vector for SSE/NEON consumption (`vp9_loopfilter.h:27`);
/// scalar callers can just read the single byte from
/// [`LoopFilterThresh`].
pub const SIMD_WIDTH: usize = 16;

/// Per-level threshold triple consumed by every loop-filter kernel
/// invocation. Mirrors libvpx `loop_filter_thresh` (the per-element
/// type of `loop_filter_info_n::lfthr[]`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct LoopFilterThresh {
    /// Macroblock-edge limit. Filter applies on the block boundary
    /// only if `|p1 - p0| + |q1 - q0| ≤ mblim`. Bigger ⇒ filter
    /// applies more often.
    pub mblim: u8,
    /// In-block edge limit. Filter applies to interior pixels only
    /// if neighbouring differences are bounded by `lim`. The
    /// "block_inside_limit" in libvpx.
    pub lim: u8,
    /// High-edge-variance threshold. Distinguishes a true edge
    /// (preserve) from a smooth gradient (smooth out). `hev_thr =
    /// filter_level >> 4`.
    pub hev_thr: u8,
}

impl LoopFilterThresh {
    /// Broadcast `mblim` 16 times — the libvpx `mblim[SIMD_WIDTH]`
    /// vector representation. Useful when wiring SIMD kernels.
    #[inline]
    #[must_use]
    pub const fn mblim_simd(&self) -> [u8; SIMD_WIDTH] {
        [self.mblim; SIMD_WIDTH]
    }

    /// Broadcast `lim` 16 times.
    #[inline]
    #[must_use]
    pub const fn lim_simd(&self) -> [u8; SIMD_WIDTH] {
        [self.lim; SIMD_WIDTH]
    }

    /// Broadcast `hev_thr` 16 times.
    #[inline]
    #[must_use]
    pub const fn hev_thr_simd(&self) -> [u8; SIMD_WIDTH] {
        [self.hev_thr; SIMD_WIDTH]
    }
}

/// Compute the threshold triple for one filter level. Mirrors libvpx
/// `update_sharpness` (`vp9_loopfilter.c:212-230`) plus the
/// hev-threshold init at `:248-249`.
///
/// `filter_level ∈ 0..=MAX_LOOP_FILTER`. Higher
/// `sharpness_level` (0..=7) tightens the in-block limit so the
/// filter is more conservative — preserving sharper edges. Mirrors
/// the parsed `sharpness_level` from the uncompressed header.
#[must_use]
pub fn compute_thresholds(filter_level: u8, sharpness_level: u8) -> LoopFilterThresh {
    let lvl = filter_level.min(MAX_LOOP_FILTER) as i32;
    let shr = sharpness_level.min(7) as i32;

    // block_inside_limit = lvl >> ((sharpness > 0) + (sharpness > 4))
    let shift = (shr > 0) as i32 + (shr > 4) as i32;
    let mut block_inside_limit = lvl >> shift;

    if shr > 0 && block_inside_limit > (9 - shr) {
        block_inside_limit = 9 - shr;
    }
    if block_inside_limit < 1 {
        block_inside_limit = 1;
    }

    // mblim = 2 * (lvl + 2) + block_inside_limit
    let mblim_val = 2 * (lvl + 2) + block_inside_limit;
    debug_assert!(mblim_val <= 255, "mblim overflow: lvl={lvl} shr={shr}");

    LoopFilterThresh {
        mblim: mblim_val as u8,
        lim: block_inside_limit as u8,
        // hev_thr = filter_level >> 4 (mirrors libvpx :249).
        hev_thr: (filter_level >> 4),
    }
}

/// Per-frame loop-filter table. Mirrors the `loop_filter_info_n::lfthr`
/// array from libvpx (`vp9_loopfilter.h`).
///
/// Indexed by filter level `0..=MAX_LOOP_FILTER` (`= 64` entries).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoopFilterInfo {
    /// One [`LoopFilterThresh`] per filter level. Entry `i` is the
    /// table for `filter_level == i`.
    pub lfthr: [LoopFilterThresh; (MAX_LOOP_FILTER as usize) + 1],
}

impl LoopFilterInfo {
    /// Build a fresh table for `sharpness_level ∈ 0..=7`. Mirrors
    /// libvpx `vp9_loop_filter_init` (`vp9_loopfilter.c:238-250`).
    #[must_use]
    pub fn new(sharpness_level: u8) -> Self {
        let mut lfthr = [LoopFilterThresh::default(); (MAX_LOOP_FILTER as usize) + 1];
        for (lvl, slot) in lfthr.iter_mut().enumerate() {
            *slot = compute_thresholds(lvl as u8, sharpness_level);
        }
        Self { lfthr }
    }

    /// Look up the threshold triple for a given filter level.
    /// Clamps `filter_level` to `MAX_LOOP_FILTER`.
    #[inline]
    #[must_use]
    pub fn thresh(&self, filter_level: u8) -> LoopFilterThresh {
        let i = filter_level.min(MAX_LOOP_FILTER) as usize;
        self.lfthr[i]
    }
}

impl Default for LoopFilterInfo {
    /// Default sharpness is `0` — matches libvpx's
    /// initial state before the first frame's header is parsed.
    fn default() -> Self {
        Self::new(0)
    }
}

// ============================================================================
// Edge-filter kernels
// ============================================================================

/// Saturating cast from `i32` to `i8`. Mirrors libvpx
/// `signed_char_clamp` (`vpx_dsp/loopfilter.c:18-20`).
#[inline]
const fn signed_char_clamp(t: i32) -> i8 {
    if t < -128 {
        -128
    } else if t > 127 {
        127
    } else {
        t as i8
    }
}

/// Mask: should we apply any filter at all to this edge? Mirrors
/// libvpx `filter_mask` (`vpx_dsp/loopfilter.c:34-46`).
///
/// Returns `0xff` if every neighbour-difference is bounded by `limit`
/// AND the macroblock-level difference is bounded by `blimit`;
/// otherwise `0`. Callers AND the resulting byte mask against their
/// per-pixel filter values.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)]
const fn filter_mask(
    limit: u8,
    blimit: u8,
    p3: u8,
    p2: u8,
    p1: u8,
    p0: u8,
    q0: u8,
    q1: u8,
    q2: u8,
    q3: u8,
) -> i8 {
    let lim = limit as i32;
    let blim = blimit as i32;
    let p0i = p0 as i32;
    let q0i = q0 as i32;
    let p1i = p1 as i32;
    let q1i = q1 as i32;

    let mut mask: i32 = 0;
    if (p3 as i32 - p2 as i32).unsigned_abs() as i32 > lim {
        mask = -1;
    }
    if (p2 as i32 - p1 as i32).unsigned_abs() as i32 > lim {
        mask = -1;
    }
    if (p1i - p0i).unsigned_abs() as i32 > lim {
        mask = -1;
    }
    if (q1i - q0i).unsigned_abs() as i32 > lim {
        mask = -1;
    }
    if (q2 as i32 - q1 as i32).unsigned_abs() as i32 > lim {
        mask = -1;
    }
    if (q3 as i32 - q2 as i32).unsigned_abs() as i32 > lim {
        mask = -1;
    }
    let mb_diff =
        (p0i - q0i).unsigned_abs() as i32 * 2 + ((p1i - q1i).unsigned_abs() as i32) / 2;
    if mb_diff > blim {
        mask = -1;
    }
    !(mask as i8)
}

/// High-edge-variance mask. Mirrors libvpx `hev_mask`
/// (`vpx_dsp/loopfilter.c:72-78`). Returns `0xff` if either of the
/// inner taps differs more than `thresh`; `0` otherwise.
#[inline]
#[must_use]
const fn hev_mask(thresh: u8, p1: u8, p0: u8, q0: u8, q1: u8) -> i8 {
    let mut hev: i32 = 0;
    if (p1 as i32 - p0 as i32).unsigned_abs() as i32 > thresh as i32 {
        hev = -1;
    }
    if (q1 as i32 - q0 as i32).unsigned_abs() as i32 > thresh as i32 {
        hev = -1;
    }
    hev as i8
}

/// Round to nearest, halves rounded up. Mirrors libvpx
/// `ROUND_POWER_OF_TWO(x, n) = (x + (1 << (n-1))) >> n` for `n=1`:
/// `(x + 1) >> 1`. Used in `filter4` outer-tap adjustment.
#[inline]
const fn round_power_of_two_1(x: i8) -> i8 {
    ((x as i32 + 1) >> 1) as i8
}

/// Core 4-tap filter applied to one transverse column-of-4 across
/// the edge. Updates `op1, op0, oq0, oq1` in place. Mirrors libvpx
/// `filter4` (`vpx_dsp/loopfilter.c:80-110`).
///
/// The libvpx implementation works in signed-char arithmetic by
/// XORing each input byte with `0x80` (flips the sign bit, mapping
/// `[0, 255]` to `[-128, 127]`); the final XOR re-maps back.
#[inline]
fn filter4(mask: i8, thresh: u8, op1: &mut u8, op0: &mut u8, oq0: &mut u8, oq1: &mut u8) {
    // Convert to centered signed via XOR-0x80 trick.
    let ps1 = (*op1 ^ 0x80) as i8 as i32;
    let ps0 = (*op0 ^ 0x80) as i8 as i32;
    let qs0 = (*oq0 ^ 0x80) as i8 as i32;
    let qs1 = (*oq1 ^ 0x80) as i8 as i32;
    let hev = hev_mask(thresh, *op1, *op0, *oq0, *oq1);

    // Outer-tap component (kept only at high-variance edges).
    let mut filter = (signed_char_clamp(ps1 - qs1) as i32) & (hev as i32);
    // Inner-tap component (applied whenever the macroblock mask is on).
    filter = (signed_char_clamp(filter + 3 * (qs0 - ps0)) as i32) & (mask as i32);

    let filter1 = (signed_char_clamp(filter + 4) as i32) >> 3;
    let filter2 = (signed_char_clamp(filter + 3) as i32) >> 3;

    *oq0 = (signed_char_clamp(qs0 - filter1) as u8) ^ 0x80;
    *op0 = (signed_char_clamp(ps0 + filter2) as u8) ^ 0x80;

    // Outer-tap adjustment (only at low-variance edges).
    let outer = (round_power_of_two_1(filter1 as i8) as i32) & !(hev as i32);

    *oq1 = (signed_char_clamp(qs1 - outer) as u8) ^ 0x80;
    *op1 = (signed_char_clamp(ps1 + outer) as u8) ^ 0x80;
}

/// 4-tap horizontal edge filter. Operates on 8 transverse columns of
/// 8 pixels each (4 above the edge, 4 below). `s` is the byte
/// immediately below the edge in column 0; `pitch` is the byte-stride
/// between rows.
///
/// Mirrors libvpx `vpx_lpf_horizontal_4_c`
/// (`vpx_dsp/loopfilter.c:112-128`).
///
/// # Safety
///
/// Pure safe code; caller must ensure `pixels[..]` contains a
/// 4-row-above + 4-row-below neighbourhood reachable by
/// `s_index + (-4..=3) * pitch + (0..8)`.
pub fn lpf_horizontal_4(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    debug_assert!(s_index >= 4 * pitch);
    for col in 0..8 {
        let s = s_index + col;
        let p3 = pixels[s - 4 * pitch];
        let p2 = pixels[s - 3 * pitch];
        let p1 = pixels[s - 2 * pitch];
        let p0 = pixels[s - pitch];
        let q0 = pixels[s];
        let q1 = pixels[s + pitch];
        let q2 = pixels[s + 2 * pitch];
        let q3 = pixels[s + 3 * pitch];
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        // SAFETY-equivalent: we split the buffer into 4 disjoint
        // single-byte slots for filter4 via repeated indexing.
        let mut op1 = pixels[s - 2 * pitch];
        let mut op0 = pixels[s - pitch];
        let mut oq0 = pixels[s];
        let mut oq1 = pixels[s + pitch];
        filter4(mask, thresh, &mut op1, &mut op0, &mut oq0, &mut oq1);
        pixels[s - 2 * pitch] = op1;
        pixels[s - pitch] = op0;
        pixels[s] = oq0;
        pixels[s + pitch] = oq1;
    }
}

/// Dual 4-tap horizontal filter — applies [`lpf_horizontal_4`] to two
/// adjacent 8-column groups with independent threshold sets. Mirrors
/// libvpx `vpx_lpf_horizontal_4_dual_c`
/// (`vpx_dsp/loopfilter.c:130-136`).
#[allow(clippy::too_many_arguments)]
pub fn lpf_horizontal_4_dual(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
) {
    lpf_horizontal_4(pixels, s_index, pitch, blimit0, limit0, thresh0);
    lpf_horizontal_4(pixels, s_index + 8, pitch, blimit1, limit1, thresh1);
}

/// 4-tap vertical edge filter. Operates on 8 rows of 8 columns each
/// (4 left of the edge, 4 right). `s` is the byte immediately to the
/// right of the edge in row 0; `pitch` is the byte-stride between rows.
///
/// Mirrors libvpx `vpx_lpf_vertical_4_c`
/// (`vpx_dsp/loopfilter.c:138-152`).
pub fn lpf_vertical_4(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    debug_assert!(s_index >= 4);
    let mut s = s_index;
    for _ in 0..8 {
        let p3 = pixels[s - 4];
        let p2 = pixels[s - 3];
        let p1 = pixels[s - 2];
        let p0 = pixels[s - 1];
        let q0 = pixels[s];
        let q1 = pixels[s + 1];
        let q2 = pixels[s + 2];
        let q3 = pixels[s + 3];
        let mask = filter_mask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        let mut op1 = pixels[s - 2];
        let mut op0 = pixels[s - 1];
        let mut oq0 = pixels[s];
        let mut oq1 = pixels[s + 1];
        filter4(mask, thresh, &mut op1, &mut op0, &mut oq0, &mut oq1);
        pixels[s - 2] = op1;
        pixels[s - 1] = op0;
        pixels[s] = oq0;
        pixels[s + 1] = oq1;
        s += pitch;
    }
}

/// Dual 4-tap vertical filter. Mirrors libvpx
/// `vpx_lpf_vertical_4_dual_c` (`vpx_dsp/loopfilter.c:154-160`).
#[allow(clippy::too_many_arguments)]
pub fn lpf_vertical_4_dual(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
) {
    lpf_vertical_4(pixels, s_index, pitch, blimit0, limit0, thresh0);
    lpf_vertical_4(pixels, s_index + 8 * pitch, pitch, blimit1, limit1, thresh1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_libvpx() {
        assert_eq!(MAX_LOOP_FILTER, 63);
        assert_eq!(SIMD_WIDTH, 16);
    }

    /// libvpx update_sharpness:
    ///   block_inside_limit = lvl >> ((sharpness > 0) + (sharpness > 4))
    /// For sharpness == 0: shift == 0; block_inside_limit = lvl
    ///   (clamped to >= 1).
    /// For lvl = 30, sharpness = 0:
    ///   block_inside_limit = 30 (>= 1).
    ///   mblim = 2*(30 + 2) + 30 = 94.
    ///   hev_thr = 30 >> 4 = 1.
    #[test]
    fn compute_thresholds_sharpness_0_lvl_30() {
        let t = compute_thresholds(30, 0);
        assert_eq!(t.mblim, 94);
        assert_eq!(t.lim, 30);
        assert_eq!(t.hev_thr, 1);
    }

    /// At lvl = 0: block_inside_limit = 0 → clamped to 1.
    ///   mblim = 2 * 2 + 1 = 5.
    ///   hev_thr = 0.
    #[test]
    fn compute_thresholds_zero_level_clamps_lim_to_1() {
        let t = compute_thresholds(0, 0);
        assert_eq!(t.mblim, 5);
        assert_eq!(t.lim, 1);
        assert_eq!(t.hev_thr, 0);
    }

    /// At sharpness = 1: shift = 1; block_inside_limit = lvl >> 1,
    ///   then clamped to <= (9 - 1) = 8.
    /// For lvl = 30, sharpness = 1:
    ///   raw = 30 >> 1 = 15; clamped to 8.
    ///   mblim = 2*(30 + 2) + 8 = 72.
    ///   hev_thr = 30 >> 4 = 1.
    #[test]
    fn compute_thresholds_sharpness_1_lvl_30() {
        let t = compute_thresholds(30, 1);
        assert_eq!(t.lim, 8);
        assert_eq!(t.mblim, 72);
        assert_eq!(t.hev_thr, 1);
    }

    /// At sharpness = 5: shift = 2; block_inside_limit = lvl >> 2,
    ///   then clamped to <= (9 - 5) = 4.
    /// For lvl = 40, sharpness = 5:
    ///   raw = 40 >> 2 = 10; clamped to 4.
    ///   mblim = 2*(40+2) + 4 = 88.
    ///   hev_thr = 40 >> 4 = 2.
    #[test]
    fn compute_thresholds_sharpness_5_lvl_40() {
        let t = compute_thresholds(40, 5);
        assert_eq!(t.lim, 4);
        assert_eq!(t.mblim, 88);
        assert_eq!(t.hev_thr, 2);
    }

    /// hev_thr saturates at lvl=63: 63 >> 4 = 3.
    #[test]
    fn hev_thr_caps_at_3_for_max_level() {
        let t = compute_thresholds(63, 0);
        assert_eq!(t.hev_thr, 3);
        assert_eq!(t.lim, 63);
        assert_eq!(t.mblim, 193); // 2 * 65 + 63
    }

    #[test]
    fn loopfilter_info_indexed_by_level() {
        let lfi = LoopFilterInfo::new(0);
        // Spot-check a few levels.
        assert_eq!(lfi.thresh(0).lim, 1);
        assert_eq!(lfi.thresh(30).mblim, 94);
        assert_eq!(lfi.thresh(63).mblim, 193);
    }

    #[test]
    fn loopfilter_info_clamps_oversize_level() {
        let lfi = LoopFilterInfo::new(0);
        // filter_level > MAX_LOOP_FILTER falls back to MAX entry.
        assert_eq!(lfi.thresh(100), lfi.thresh(MAX_LOOP_FILTER));
    }

    #[test]
    fn simd_broadcast_helpers() {
        let t = LoopFilterThresh {
            mblim: 100,
            lim: 30,
            hev_thr: 2,
        };
        assert_eq!(t.mblim_simd(), [100; SIMD_WIDTH]);
        assert_eq!(t.lim_simd(), [30; SIMD_WIDTH]);
        assert_eq!(t.hev_thr_simd(), [2; SIMD_WIDTH]);
    }

    /// Sharpness clamping: a `sharpness_level > 7` should be treated
    /// like `7` (the libvpx field is a 3-bit literal so 7 is the
    /// max). Verify our `.min(7)` clamp.
    #[test]
    fn sharpness_clamps_at_7() {
        assert_eq!(compute_thresholds(30, 7), compute_thresholds(30, 8));
        assert_eq!(compute_thresholds(30, 7), compute_thresholds(30, 100));
    }

    /// `LoopFilterInfo::default()` matches `LoopFilterInfo::new(0)`.
    #[test]
    fn default_is_sharpness_zero() {
        assert_eq!(LoopFilterInfo::default(), LoopFilterInfo::new(0));
    }

    // ---- Edge-filter kernels ----

    /// Constant-input across an edge: filter has nothing to do.
    /// With pixels all = 128 across the 8 neighbour-rows, mask is
    /// trivially "apply", but filter4 sees `qs0 - ps0 == 0` and
    /// `ps1 - qs1 == 0`, so all updates are zero. The block is
    /// returned unchanged.
    #[test]
    fn lpf_horizontal_4_constant_input_unchanged() {
        // 8-row tall buffer, 8 columns wide, pitch = 8.
        let mut pixels = [128u8; 8 * 8];
        let original = pixels;
        lpf_horizontal_4(&mut pixels, 4 * 8, 8, 20, 10, 4);
        assert_eq!(pixels, original);
    }

    /// Same for vertical.
    #[test]
    fn lpf_vertical_4_constant_input_unchanged() {
        let mut pixels = [128u8; 8 * 8];
        let original = pixels;
        lpf_vertical_4(&mut pixels, 4, 8, 20, 10, 4);
        assert_eq!(pixels, original);
    }

    /// If `limit` is `0`, every neighbour difference > 0 fails the
    /// mask check ⇒ filter mask becomes 0 ⇒ no pixels are touched.
    /// We construct an input with non-zero neighbour differences and
    /// verify the buffer is unchanged.
    #[test]
    fn lpf_horizontal_4_mask_blocks_filter_when_limit_zero() {
        let mut pixels = [0u8; 8 * 8];
        // 8 rows, pitch 8: rows 0..7. Each column has a unique
        // gradient that creates a real edge between row 3 (p0) and
        // row 4 (q0).
        for r in 0..8 {
            for c in 0..8 {
                pixels[r * 8 + c] = if r < 4 { 100 } else { 150 };
            }
        }
        let before = pixels;
        lpf_horizontal_4(&mut pixels, 4 * 8, 8, /*blimit*/ 0, /*limit*/ 0, /*thresh*/ 0);
        // limit=0, blimit=0: filter_mask returns 0 because the
        // |p0 - q0| * 2 = 100 > blimit = 0.
        assert_eq!(pixels, before);
    }

    /// True step edge: 100 in `p`-rows, 150 in `q`-rows, with
    /// permissive thresholds. The 4-tap filter smooths the boundary
    /// slightly.
    #[test]
    fn lpf_horizontal_4_smooths_step_edge() {
        let mut pixels = [0u8; 8 * 8];
        for r in 0..8 {
            for c in 0..8 {
                pixels[r * 8 + c] = if r < 4 { 100 } else { 150 };
            }
        }
        // Wide-enough thresholds: limit=20 (every interior neighbour
        // pair differs by 0, well under the limit). blimit must
        // allow |p0-q0|*2 + |p1-q1|/2 = 100 + 25 = 125. Use 200 to
        // be safely above.
        let before = pixels;
        lpf_horizontal_4(&mut pixels, 4 * 8, 8, /*blimit*/ 200, /*limit*/ 20, /*thresh*/ 4);
        // Edge pixels should change.
        assert_ne!(pixels, before);
        // Symmetry: column 0 vs column 7 should be identical since
        // the input is column-uniform.
        for r in 0..8 {
            assert_eq!(pixels[r * 8], pixels[r * 8 + 7]);
        }
        // The step should narrow: the new p0 row should be larger
        // than 100 (pulled up), and the new q0 row smaller than 150
        // (pulled down).
        assert!(pixels[3 * 8] > 100, "new p0 should be > 100");
        assert!(pixels[4 * 8] < 150, "new q0 should be < 150");
    }

    /// Vertical filter is functionally identical to horizontal — by
    /// transposing the data layout we should get the same numeric
    /// result. This pins implementation consistency.
    #[test]
    fn lpf_vertical_4_smooths_step_edge_consistent_with_horizontal() {
        // Horizontal: row-by-row 100/150 step.
        let mut h_pixels = [0u8; 8 * 8];
        for r in 0..8 {
            for c in 0..8 {
                h_pixels[r * 8 + c] = if r < 4 { 100 } else { 150 };
            }
        }
        lpf_horizontal_4(&mut h_pixels, 4 * 8, 8, 200, 20, 4);

        // Vertical: column-by-column 100/150 step (transpose of above).
        let mut v_pixels = [0u8; 8 * 8];
        for r in 0..8 {
            for c in 0..8 {
                v_pixels[r * 8 + c] = if c < 4 { 100 } else { 150 };
            }
        }
        lpf_vertical_4(&mut v_pixels, 4, 8, 200, 20, 4);

        // Verify v_pixels is the transpose of h_pixels.
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    h_pixels[r * 8 + c],
                    v_pixels[c * 8 + r],
                    "mismatch at h({r},{c}) vs v({c},{r})"
                );
            }
        }
    }

    /// `_dual` variants apply the single-edge filter to the two
    /// halves of the input with independent thresholds. We check
    /// that running `_dual` with `(t0, t0)` is identical to running
    /// the single-edge filter on each half.
    #[test]
    fn lpf_horizontal_4_dual_matches_two_singles() {
        let mut a = [0u8; 16 * 8];
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = (i as u8).wrapping_mul(3);
        }
        let mut b = a;

        lpf_horizontal_4_dual(&mut a, 4 * 8, 8, 200, 20, 4, 200, 20, 4);

        lpf_horizontal_4(&mut b, 4 * 8, 8, 200, 20, 4);
        lpf_horizontal_4(&mut b, 4 * 8 + 8, 8, 200, 20, 4);

        assert_eq!(a, b);
    }
}
