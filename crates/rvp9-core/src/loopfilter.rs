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
//! ## Deferred
//!
//! - The 12 edge-filter kernels (`vpx_lpf_{vertical,horizontal}_{4,8,16}`,
//!   with `_dual` variants) — Task 6.2.
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
}
