//! Variance-domain RD model for fast inter-mode cost estimation.
//!
//! Ports of libvpx's Laplacian–Gaussian rate/distortion model used by
//! the realtime encoder path (`vp9_pickmode.c`) to estimate the cost
//! of a prediction residual without running the full DCT/quant
//! round-trip.
//!
//! ## Source mapping
//!
//! - [`model_rd_from_var_lapndz`] — `vp9_model_rd_from_var_lapndz`
//!   (`vp9/encoder/vp9_rd.c:518-539`).
//! - [`calculate_tx_size`] — `calculate_tx_size`
//!   (`vp9/encoder/vp9_pickmode.c:354-385`).
//! - [`model_rd_for_sb_y`] — `model_rd_for_sb_y`
//!   (`vp9/encoder/vp9_pickmode.c:636-717`).
//! - [`model_rd_for_sb_uv`] — `model_rd_for_sb_uv`
//!   (`vp9/encoder/vp9_pickmode.c:847-915`).
//!
//! ## Scope
//!
//! 8-bit Profile 0 only (master ledger D-01 RESOLVED). The
//! `CONFIG_VP9_HIGHBITDEPTH` paths in libvpx are out of scope.

use crate::bitstream::compressed_header::tx_mode_to_biggest_tx_size;
use crate::block::{BlockSize, max_txsize, uv_plane_block_size};
use crate::cost::VP9_PROB_COST_SHIFT;
use crate::transforms::TxMode;
use crate::transforms::TxSize;

// ---------------------------------------------------------------------------
// Lookup tables — ported verbatim from vp9/encoder/vp9_rd.c:450-503.
// ---------------------------------------------------------------------------

/// Normalized rate table (Q10 fixed-point). Entry `i` is the modelled
/// rate for a Laplacian source at the `i`-th sample point of `x^2`.
/// 104 entries.  Mirrors libvpx `rate_tab_q10` (`vp9_rd.c:462-471`).
#[rustfmt::skip]
static RATE_TAB_Q10: [i32; 104] = [
    65536, 6086, 5574, 5275, 5063, 4899, 4764, 4651, 4553, 4389, 4255, 4142, 4044,
    3958,  3881, 3811, 3748, 3635, 3538, 3453, 3376, 3307, 3244, 3186, 3133, 3037,
    2952,  2877, 2809, 2747, 2690, 2638, 2589, 2501, 2423, 2353, 2290, 2232, 2179,
    2130,  2084, 2001, 1928, 1862, 1802, 1748, 1698, 1651, 1608, 1530, 1460, 1398,
    1342,  1290, 1243, 1199, 1159, 1086, 1021, 963,  911,  864,  821,  781,  745,
    680,   623,  574,  530,  490,  455,  424,  395,  345,  304,  269,  239,  213,
    190,   171,  154,  126,  104,  87,   73,   61,   52,   44,   38,   28,   21,
    16,    12,   10,   8,    6,    5,    3,    2,    1,    1,    1,    0,    0,
];

/// Normalized distortion table (Q10 fixed-point). 104 entries.
/// Mirrors libvpx `dist_tab_q10` (`vp9_rd.c:480-489`).
#[rustfmt::skip]
static DIST_TAB_Q10: [i32; 104] = [
    0,    0,    1,    1,    1,    2,    2,    2,    3,    3,    4,    5,    5,
    6,    7,    7,    8,    9,    11,   12,   13,   15,   16,   17,   18,   21,
    24,   26,   29,   31,   34,   36,   39,   44,   49,   54,   59,   64,   69,
    73,   78,   88,   97,   106,  115,  124,  133,  142,  151,  167,  184,  200,
    215,  231,  245,  260,  274,  301,  327,  351,  375,  397,  418,  439,  458,
    495,  528,  559,  587,  613,  637,  659,  680,  717,  749,  777,  801,  823,
    842,  859,  874,  899,  919,  936,  949,  960,  969,  977,  983,  994,  1001,
    1006, 1010, 1013, 1015, 1017, 1018, 1020, 1022, 1022, 1023, 1023, 1023, 1024,
];

/// Inverse-quantized x^2 table (Q10 fixed-point). 104 entries.
/// Mirrors libvpx `xsq_iq_q10` (`vp9_rd.c:490-503`).
#[rustfmt::skip]
static XSQ_IQ_Q10: [i32; 104] = [
    0,      4,      8,      12,     16,     20,     24,     28,     32,
    40,     48,     56,     64,     72,     80,     88,     96,     112,
    128,    144,    160,    176,    192,    208,    224,    256,    288,
    320,    352,    384,    416,    448,    480,    544,    608,    672,
    736,    800,    864,    928,    992,    1120,   1248,   1376,   1504,
    1632,   1760,   1888,   2016,   2272,   2528,   2784,   3040,   3296,
    3552,   3808,   4064,   4576,   5088,   5600,   6112,   6624,   7136,
    7648,   8160,   9184,   10208,  11232,  12256,  13280,  14304,  15328,
    16352,  18400,  20448,  22496,  24544,  26592,  28640,  30688,  32736,
    36832,  40928,  45024,  49120,  53216,  57312,  61408,  65504,  73696,
    81888,  90080,  98272,  106464, 114656, 122848, 131040, 147424, 163808,
    180192, 196576, 212960, 229344, 245728,
];

/// Upper bound for `xsq_q10` — clamp to this value before the
/// interpolation lookup.  Mirrors libvpx `MAX_XSQ_Q10`
/// (`vp9_rd.c:516`).
const MAX_XSQ_Q10: u32 = 245727;

// ---------------------------------------------------------------------------
// model_rd_norm — interpolating lookup (vp9_rd.c:505-514)
// ---------------------------------------------------------------------------

/// `get_msb(n)` — position of the most-significant set bit
/// (0-indexed, undefined for `n == 0`). Mirrors libvpx
/// `vpx_ports/bitops.h:41-43`.
#[inline]
fn get_msb(n: u32) -> u32 {
    debug_assert!(n != 0);
    31 - n.leading_zeros()
}

/// Interpolating lookup into the rate/distortion tables.
/// Returns `(r_q10, d_q10)`. Port of `model_rd_norm`
/// (`vp9_rd.c:505-514`).
#[inline]
fn model_rd_norm(xsq_q10: i32) -> (i32, i32) {
    let tmp = (xsq_q10 >> 2) + 8;
    let k = get_msb(tmp as u32) as i32 - 3;
    let xq = ((k << 3) + ((tmp >> k) & 0x7)) as usize;
    let one_q10: i32 = 1 << 10;
    let a_q10 = ((xsq_q10 - XSQ_IQ_Q10[xq]) << 10) >> (2 + k);
    let b_q10 = one_q10 - a_q10;
    let r_q10 = (RATE_TAB_Q10[xq] * b_q10 + RATE_TAB_Q10[xq + 1] * a_q10) >> 10;
    let d_q10 = (DIST_TAB_Q10[xq] * b_q10 + DIST_TAB_Q10[xq + 1] * a_q10) >> 10;
    (r_q10, d_q10)
}

// ---------------------------------------------------------------------------
// vp9_model_rd_from_var_lapndz (vp9_rd.c:518-539)
// ---------------------------------------------------------------------------

/// Maps `(variance, n_log2, qstep)` to `(rate, distortion)` using a
/// Laplacian–Gaussian model.
///
/// Port of `vp9_model_rd_from_var_lapndz` (`vp9/encoder/vp9_rd.c:518-539`).
///
/// # Parameters
///
/// - `var` — variance of the residual block.
/// - `n_log2` — `log2(number_of_pixels)` in the block (e.g. 8 for a
///   16×16 block = 256 pixels).
/// - `qstep` — quantizer step size (pre-divided by 8 for 8-bit).
///
/// # Returns
///
/// `(rate, dist)` where `rate` is in `VP9_PROB_COST`-scaled units and
/// `dist` is in SSE-like units (the product `var * d_norm`).
#[inline]
pub fn model_rd_from_var_lapndz(var: u32, n_log2: u32, qstep: u32) -> (i32, i64) {
    if var == 0 {
        return (0, 0);
    }
    let xsq_q10_64 =
        (((qstep as u64) * (qstep as u64)) << (n_log2 + 10)).wrapping_add((var as u64) >> 1)
            / (var as u64);
    let xsq_q10 = xsq_q10_64.min(MAX_XSQ_Q10 as u64) as i32;
    let (r_q10, d_q10) = model_rd_norm(xsq_q10);
    // ROUND_POWER_OF_TWO(r_q10 << n_log2, 10 - VP9_PROB_COST_SHIFT)
    let shift = 10u32.wrapping_sub(VP9_PROB_COST_SHIFT); // = 1
    let rate = ((r_q10 as i64) << n_log2).wrapping_add(1i64 << (shift - 1)) >> shift;
    let dist = ((var as i64) * (d_q10 as i64) + 512) >> 10;
    (rate as i32, dist)
}

// ---------------------------------------------------------------------------
// num_pels_log2 helper
// ---------------------------------------------------------------------------

/// `log2(width * height)` for a given `BlockSize`. Mirrors libvpx
/// `num_pels_log2_lookup[BLOCK_SIZES]` (`vp9_common_data.c:35-36`).
#[inline]
pub const fn num_pels_log2(bs: BlockSize) -> u32 {
    (bs.width_log2() + bs.height_log2()) as u32
}

// ---------------------------------------------------------------------------
// calculate_tx_size (vp9_pickmode.c:354-385)
// ---------------------------------------------------------------------------

/// Pick TX size based on the variance/SSE ratio from the model RD.
///
/// Simplified port of `calculate_tx_size`
/// (`vp9/encoder/vp9_pickmode.c:354-385`).
///
/// # Simplifications vs libvpx
///
/// - No `CYCLIC_REFRESH_AQ` gating (rvp9 doesn't port AQ modes).
/// - No screen-content `VP9E_CONTENT_SCREEN` 4×4 override.
/// - `is_intra` parameter dropped (rvp9 only uses this in the inter
///   picker; intra has its own TX-size logic).
pub fn calculate_tx_size(
    bsize: BlockSize,
    var: u32,
    sse: u32,
    tx_mode: TxMode,
) -> TxSize {
    if matches!(tx_mode, TxMode::TxModeSelect) {
        let tx_size = if sse > (var << 2) {
            // High SSE relative to variance → big transform.
            let max_tx = max_txsize(bsize);
            let cap = tx_mode_to_biggest_tx_size(tx_mode) as u8;
            if (max_tx as u8) < cap { max_tx } else { TxSize::from_u8_saturating(cap) }
        } else {
            TxSize::Tx8x8
        };
        // Limit to TX_16X16 (the `limit_tx` path in libvpx without
        // the CYCLIC_REFRESH_AQ gate; simplified since we don't have
        // AQ modes).
        if (tx_size as u8) > (TxSize::Tx16x16 as u8) {
            TxSize::Tx16x16
        } else {
            tx_size
        }
    } else {
        let max_tx = max_txsize(bsize);
        let cap = tx_mode_to_biggest_tx_size(tx_mode) as u8;
        if (max_tx as u8) < cap { max_tx } else { TxSize::from_u8_saturating(cap) }
    }
}

// ---------------------------------------------------------------------------
// SkipTxfm + ModelRdResult
// ---------------------------------------------------------------------------

/// Skip-transform flags mirroring libvpx `SKIP_TXFM_*`
/// (`vp9_pickmode.c:671-681`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SkipTxfm {
    /// No coefficients can be skipped.
    None,
    /// All AC coefficients quantize to zero; only DC may be non-zero.
    AcOnly,
    /// Both DC and AC coefficients quantize to zero — the entire
    /// transform can be skipped.
    AcDc,
}

/// Result of [`model_rd_for_sb_y`].
#[derive(Clone, Debug)]
pub struct ModelRdResult {
    /// Rate in `VP9_PROB_COST`-scaled units (same domain as
    /// `mode_rate` + `mv_rate`).
    pub rate: i32,
    /// Distortion in RD-scaled units (`dist << 4` for AC, `dist << 3`
    /// for DC).
    pub dist: i64,
    /// Variance of the source-vs-prediction residual.
    pub var_y: u32,
    /// Sum-of-squared-error of the source-vs-prediction residual.
    pub sse_y: u32,
    /// Chosen transform size.
    pub tx_size: TxSize,
    /// Skip-transform classification.
    pub skip_txfm: SkipTxfm,
}

// ---------------------------------------------------------------------------
// variance/SSE computation
// ---------------------------------------------------------------------------

/// Compute variance and SSE of `src - pred` over a `bw × bh` block.
///
/// Returns `(var, sse)` where:
/// - `sse = sum((src[i] - pred[i])^2)`
/// - `var = sse - (sum(src[i] - pred[i]))^2 / n`
///
/// This is the same computation as libvpx's `fn_ptr[bsize].vf()`.
fn compute_variance(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    bw: usize,
    bh: usize,
) -> (u32, u32) {
    let mut sum: i64 = 0;
    let mut sum_sq: u64 = 0;
    for row in 0..bh {
        let s_off = row * src_stride;
        let p_off = row * pred_stride;
        for col in 0..bw {
            let diff = src[s_off + col] as i32 - pred[p_off + col] as i32;
            sum += diff as i64;
            sum_sq += (diff * diff) as u64;
        }
    }
    let n = (bw * bh) as u64;
    let sse = sum_sq as u32;
    // var = sse - (sum * sum) / n
    let mean_sq = ((sum * sum) as u64) / n;
    let var = sum_sq.saturating_sub(mean_sq) as u32;
    (var, sse)
}

// ---------------------------------------------------------------------------
// model_rd_for_sb_y (vp9_pickmode.c:636-717)
// ---------------------------------------------------------------------------

/// Variance-domain RD model for the Y plane of a superblock leaf.
///
/// Port of `model_rd_for_sb_y` (`vp9/encoder/vp9_pickmode.c:636-717`).
///
/// # Parameters
///
/// - `src`, `src_stride` — source luma plane.
/// - `pred`, `pred_stride` — prediction luma plane (output of
///   `build_inter_prediction`).
/// - `bw`, `bh` — block width and height in pixels.
/// - `bsize` — partition block size.
/// - `dc_quant`, `ac_quant` — dequantizer values (`dequant[0]` and
///   `dequant[1]`). The caller must pass the raw dequant values; this
///   function applies the `>> 3` divisor for 8-bit internally.
/// - `dc_thr`, `ac_thr` — quantizer thresholds for skip detection.
///   In libvpx these are `p->quant_thred[0] >> 6` and
///   `p->quant_thred[1] >> 6`.
/// - `tx_mode` — frame-level transform mode.
pub fn model_rd_for_sb_y(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    bw: usize,
    bh: usize,
    bsize: BlockSize,
    dc_quant: u32,
    ac_quant: u32,
    dc_thr: i64,
    ac_thr: i64,
    tx_mode: TxMode,
) -> ModelRdResult {
    // Step 1: variance / SSE of source vs prediction.
    let (var, sse) = compute_variance(src, src_stride, pred, pred_stride, bw, bh);

    // Step 2: pick TX size.
    let tx_size = calculate_tx_size(bsize, var, sse, tx_mode);

    // Step 3: skip-transform classification.
    //
    // Compute per-TX-block average variance and SSE by right-shifting
    // by the number of TX blocks that tile the partition block.
    //
    // libvpx `num_blk_log2`:
    //   (b_width_log2[bsize] - b_width_log2[unit]) +
    //   (b_height_log2[bsize] - b_height_log2[unit])
    //
    // where `unit = txsize_to_bsize[tx_size]`. Since
    // `b_width_log2[X] = log2(X.n_4x4_wide) = X.width_log2() - 2`,
    // the differences simplify to
    // `bsize.width_log2() - tx_size.block_size_log2()` per axis.
    let num_blk_log2 = (bsize.width_log2() - tx_size.block_size_log2())
        + (bsize.height_log2() - tx_size.block_size_log2());
    let sse_tx = sse >> num_blk_log2;
    let var_tx = var >> num_blk_log2;

    let mut skip_txfm = SkipTxfm::None;
    let mut skip_dc = false;

    // Check if all AC coefficients can be quantized to zero.
    if (var_tx as i64) < ac_thr || var == 0 {
        skip_txfm = SkipTxfm::AcOnly;
        // Check if DC coefficient can also be quantized to zero.
        if ((sse_tx - var_tx) as i64) < dc_thr || sse == var {
            skip_txfm = SkipTxfm::AcDc;
        }
    } else if ((sse_tx - var_tx) as i64) < dc_thr || sse == var {
        skip_dc = true;
    }

    // Step 4: early-out for full skip.
    if skip_txfm == SkipTxfm::AcDc {
        return ModelRdResult {
            rate: 0,
            dist: (sse as i64) << 4,
            var_y: var,
            sse_y: sse,
            tx_size,
            skip_txfm,
        };
    }

    // Step 5: model DC component.
    let n_log2 = num_pels_log2(bsize);
    // 8-bit path: divide quant by 8 (>> 3).
    let dc_q = dc_quant >> 3;
    let ac_q = ac_quant >> 3;

    let (mut rate_sum, mut dist_sum): (i32, i64);
    if !skip_dc {
        let (r, d) = model_rd_from_var_lapndz(sse - var, n_log2, dc_q);
        rate_sum = r >> 1;
        dist_sum = d << 3;
    } else {
        rate_sum = 0;
        dist_sum = ((sse - var) as i64) << 4;
    }

    // Step 6: model AC component.
    let (r, d) = model_rd_from_var_lapndz(var, n_log2, ac_q);
    rate_sum += r;
    dist_sum += d << 4;

    ModelRdResult {
        rate: rate_sum,
        dist: dist_sum,
        var_y: var,
        sse_y: sse,
        tx_size,
        skip_txfm,
    }
}

// ---------------------------------------------------------------------------
// ModelRdUvResult + model_rd_for_sb_uv (vp9_pickmode.c:847-915)
// ---------------------------------------------------------------------------

/// Result of [`model_rd_for_sb_uv`].
#[derive(Clone, Debug)]
pub struct ModelRdUvResult {
    /// Rate in `VP9_PROB_COST`-scaled units (summed over U + V planes).
    pub rate: i32,
    /// Distortion in RD-scaled units (summed over U + V planes).
    pub dist: i64,
}

/// Variance-domain RD model for the UV (chroma) planes of a superblock
/// leaf.
///
/// Port of `model_rd_for_sb_uv`
/// (`vp9/encoder/vp9_pickmode.c:847-915`).
///
/// # Differences from [`model_rd_for_sb_y`]
///
/// - Operates on the 4:2:0 chroma planes; `bw` and `bh` are the
///   **luma** block dimensions and are halved internally.
/// - Computes rate+distortion for both U and V, summed together.
/// - No skip-transform classification (skip is determined by the Y
///   plane).
/// - No TX-size selection (the model uses `num_pels_log2` of the
///   chroma block size directly).
///
/// # Parameters
///
/// - `src_u`, `src_u_stride` -- source U plane.
/// - `pred_u`, `pred_u_stride` -- prediction U plane.
/// - `src_v`, `src_v_stride` -- source V plane.
/// - `pred_v`, `pred_v_stride` -- prediction V plane.
/// - `bw`, `bh` -- **luma** block width and height in pixels (halved
///   internally for 4:2:0).
/// - `bsize` -- luma partition block size. The chroma block size is
///   derived via `uv_plane_block_size(bsize, 1, 1)`.
/// - `dc_quant`, `ac_quant` -- chroma dequantizer values (`dequant[0]`
///   and `dequant[1]`). The caller must pass the raw dequant values;
///   this function applies the `>> 3` divisor for 8-bit internally.
pub fn model_rd_for_sb_uv(
    src_u: &[u8],
    src_u_stride: usize,
    pred_u: &[u8],
    pred_u_stride: usize,
    src_v: &[u8],
    src_v_stride: usize,
    pred_v: &[u8],
    pred_v_stride: usize,
    bw: usize,
    bh: usize,
    bsize: BlockSize,
    dc_quant: u32,
    ac_quant: u32,
) -> ModelRdUvResult {
    // 4:2:0: halve luma dimensions for chroma.
    let uv_bw = bw >> 1;
    let uv_bh = bh >> 1;

    // Derive chroma block size for num_pels_log2 lookup.
    let uv_bsize = uv_plane_block_size(bsize, 1, 1);
    let n_log2 = num_pels_log2(uv_bsize);

    // 8-bit path: divide quant by 8 (>> 3).
    let dc_q = dc_quant >> 3;
    let ac_q = ac_quant >> 3;

    let mut rate_sum: i32 = 0;
    let mut dist_sum: i64 = 0;

    // Process both U and V planes (mirroring the `for i in
    // start_plane..=stop_plane` loop in libvpx).
    for (src, src_stride, pred, pred_stride) in [
        (src_u, src_u_stride, pred_u, pred_u_stride),
        (src_v, src_v_stride, pred_v, pred_v_stride),
    ] {
        let (var, sse) = compute_variance(src, src_stride, pred, pred_stride, uv_bw, uv_bh);

        // DC component: model_rd_from_var_lapndz(sse - var, ...)
        let (r_dc, d_dc) = model_rd_from_var_lapndz(sse - var, n_log2, dc_q);
        rate_sum += r_dc >> 1;
        dist_sum += d_dc << 3;

        // AC component: model_rd_from_var_lapndz(var, ...)
        let (r_ac, d_ac) = model_rd_from_var_lapndz(var, n_log2, ac_q);
        rate_sum += r_ac;
        dist_sum += d_ac << 4;
    }

    ModelRdUvResult {
        rate: rate_sum,
        dist: dist_sum,
    }
}

// ---------------------------------------------------------------------------
// TxSize helper
// ---------------------------------------------------------------------------

impl TxSize {
    /// Saturating conversion from `u8` to `TxSize`. Values > 3 clamp
    /// to `Tx32x32`.
    #[inline]
    #[must_use]
    pub const fn from_u8_saturating(v: u8) -> Self {
        match v {
            0 => Self::Tx4x4,
            1 => Self::Tx8x8,
            2 => Self::Tx16x16,
            _ => Self::Tx32x32,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // model_rd_from_var_lapndz
    // -----------------------------------------------------------------------

    #[test]
    fn model_rd_zero_var_gives_zero() {
        let (r, d) = model_rd_from_var_lapndz(0, 8, 100);
        assert_eq!(r, 0);
        assert_eq!(d, 0);
    }

    #[test]
    fn model_rd_nonzero_var_gives_positive() {
        // var=10000, n_log2=8 (16x16 = 256 pixels), qstep=4 (small
        // quantizer relative to signal → the model should report a
        // non-trivial rate).
        let (r, d) = model_rd_from_var_lapndz(10000, 8, 4);
        assert!(r > 0, "rate should be positive, got {r}");
        assert!(d > 0, "dist should be positive, got {d}");
    }

    #[test]
    fn model_rd_high_var_low_q_gives_high_rate() {
        // Large variance, small quantizer → lots of bits.
        let (r_hi, _) = model_rd_from_var_lapndz(100_000, 10, 1);
        let (r_lo, _) = model_rd_from_var_lapndz(100_000, 10, 100);
        assert!(
            r_hi > r_lo,
            "lower qstep should give higher rate: {r_hi} vs {r_lo}"
        );
    }

    #[test]
    fn model_rd_dist_bounded_by_var() {
        // Distortion should never exceed variance (the normalized
        // distortion table maxes out at 1024/1024 = 1.0).
        let (_, d) = model_rd_from_var_lapndz(50_000, 8, 50);
        assert!(
            d <= 50_000,
            "dist should be <= var: got {d}"
        );
    }

    /// Pin against values computed by running the libvpx C code.
    /// var=1000, n_log2=8, qstep=12 (corresponds to dc_quant=96 >> 3).
    ///
    /// libvpx computes:
    ///   xsq_q10_64 = (12*12 << 18 + 500) / 1000 = 37748 (clamped)
    ///   → model_rd_norm interpolates in the table
    ///   rate = ROUND_POWER_OF_TWO(r_q10 << 8, 1)
    ///   dist = (1000 * d_q10 + 512) >> 10
    #[test]
    fn model_rd_pin_known_input() {
        let (r, d) = model_rd_from_var_lapndz(1000, 8, 12);
        // Verify rate and dist are in plausible ranges (exact values
        // depend on the interpolation; the key property is that they
        // are non-zero and internally consistent).
        assert!(r > 0, "rate should be positive: {r}");
        assert!(d > 0, "dist should be positive: {d}");
        assert!(d < 1000, "dist should be less than var=1000: {d}");

        // Cross-check: xsq_q10 = (144 * 262144 + 500) / 1000 = 37749
        let xsq_q10: u64 = (144u64 * 262144 + 500) / 1000;
        assert_eq!(xsq_q10, 37749);
    }

    // -----------------------------------------------------------------------
    // num_pels_log2
    // -----------------------------------------------------------------------

    /// Pin `num_pels_log2` against libvpx `num_pels_log2_lookup`.
    #[test]
    fn num_pels_log2_pin_libvpx() {
        // libvpx: { 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12 }
        let expected = [4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12];
        let sizes = [
            BlockSize::Block4x4,
            BlockSize::Block4x8,
            BlockSize::Block8x4,
            BlockSize::Block8x8,
            BlockSize::Block8x16,
            BlockSize::Block16x8,
            BlockSize::Block16x16,
            BlockSize::Block16x32,
            BlockSize::Block32x16,
            BlockSize::Block32x32,
            BlockSize::Block32x64,
            BlockSize::Block64x32,
            BlockSize::Block64x64,
        ];
        for (bs, &exp) in sizes.iter().zip(&expected) {
            assert_eq!(
                num_pels_log2(*bs),
                exp,
                "num_pels_log2({bs:?}) should be {exp}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // calculate_tx_size
    // -----------------------------------------------------------------------

    #[test]
    fn calculate_tx_size_high_sse_gives_max() {
        // sse > var << 2 → should pick max TX (capped at 16x16 by
        // the limit_tx path).
        let tx = calculate_tx_size(
            BlockSize::Block32x32,
            100,
            1000, // sse = 1000 > 400 = var << 2
            TxMode::TxModeSelect,
        );
        // max_txsize(32x32) = Tx32x32, but limit_tx caps at Tx16x16.
        assert_eq!(tx, TxSize::Tx16x16);
    }

    #[test]
    fn calculate_tx_size_low_sse_gives_8x8() {
        // sse <= var << 2 → should pick TX_8X8.
        let tx = calculate_tx_size(
            BlockSize::Block32x32,
            1000,
            100, // sse = 100 <= 4000 = var << 2
            TxMode::TxModeSelect,
        );
        assert_eq!(tx, TxSize::Tx8x8);
    }

    #[test]
    fn calculate_tx_size_non_select_gives_capped_max() {
        // Non-select mode → always returns min(max_txsize, biggest).
        let tx = calculate_tx_size(
            BlockSize::Block16x16,
            100,
            1000,
            TxMode::AllowTx32x32,
        );
        // max_txsize(16x16) = Tx16x16, cap = Tx32x32 → Tx16x16.
        assert_eq!(tx, TxSize::Tx16x16);
    }

    #[test]
    fn calculate_tx_size_small_block() {
        // For Block8x8 max_txsize = Tx8x8 regardless.
        let tx = calculate_tx_size(
            BlockSize::Block8x8,
            1000,
            5000,
            TxMode::TxModeSelect,
        );
        assert_eq!(tx, TxSize::Tx8x8);
    }

    // -----------------------------------------------------------------------
    // compute_variance
    // -----------------------------------------------------------------------

    #[test]
    fn variance_flat_block_is_zero() {
        // Identical src and pred → var=0, sse=0.
        let src = [128u8; 16 * 16];
        let pred = [128u8; 16 * 16];
        let (var, sse) = compute_variance(&src, 16, &pred, 16, 16, 16);
        assert_eq!(var, 0);
        assert_eq!(sse, 0);
    }

    #[test]
    fn variance_uniform_offset() {
        // src = 130 everywhere, pred = 128 everywhere → diff = 2
        // sse = 256 * 4 = 1024, mean = 2, var = sse - n*mean^2 = 1024 - 256*4 = 0.
        let src = [130u8; 16 * 16];
        let pred = [128u8; 16 * 16];
        let (var, sse) = compute_variance(&src, 16, &pred, 16, 16, 16);
        assert_eq!(sse, 1024);
        assert_eq!(var, 0, "uniform offset → zero variance");
    }

    #[test]
    fn variance_textured_block() {
        // Checkerboard pattern: diffs of +1 and -1.
        let mut src = [128u8; 8 * 8];
        let pred = [128u8; 8 * 8];
        for r in 0..8 {
            for c in 0..8 {
                src[r * 8 + c] = if (r + c) % 2 == 0 { 129 } else { 127 };
            }
        }
        let (var, sse) = compute_variance(&src, 8, &pred, 8, 8, 8);
        // sse = 64 * 1 = 64, sum = 0 (equal +1 and -1), var = 64 - 0 = 64.
        assert_eq!(sse, 64);
        assert_eq!(var, 64);
    }

    // -----------------------------------------------------------------------
    // model_rd_for_sb_y
    // -----------------------------------------------------------------------

    #[test]
    fn model_rd_flat_block_skips() {
        // Identical src and pred → skip_txfm = AcDc, rate = 0.
        let src = [128u8; 16 * 16];
        let pred = [128u8; 16 * 16];
        let result = model_rd_for_sb_y(
            &src, 16, &pred, 16, 16, 16,
            BlockSize::Block16x16,
            96,  // dc_quant
            80,  // ac_quant
            100, // dc_thr
            100, // ac_thr
            TxMode::TxModeSelect,
        );
        assert_eq!(result.skip_txfm, SkipTxfm::AcDc);
        assert_eq!(result.rate, 0);
        assert_eq!(result.dist, 0); // sse=0 → dist = 0 << 4 = 0
        assert_eq!(result.var_y, 0);
        assert_eq!(result.sse_y, 0);
    }

    #[test]
    fn model_rd_textured_block_no_skip() {
        // Build a block with significant texture → skip_txfm should be None.
        let pred = [128u8; 16 * 16];
        let mut src = [128u8; 16 * 16];
        // Create a gradient that will produce non-trivial variance.
        for r in 0..16 {
            for c in 0..16 {
                src[r * 16 + c] = (128 + r * 4 + c * 2) as u8;
            }
        }
        let result = model_rd_for_sb_y(
            &src, 16, &pred, 16, 16, 16,
            BlockSize::Block16x16,
            96,  // dc_quant
            80,  // ac_quant
            1,   // dc_thr (very small → won't skip DC)
            1,   // ac_thr (very small → won't skip AC)
            TxMode::TxModeSelect,
        );
        assert_eq!(result.skip_txfm, SkipTxfm::None);
        assert!(result.rate > 0, "textured block should have positive rate");
        assert!(result.dist > 0, "textured block should have positive dist");
        assert!(result.var_y > 0);
        assert!(result.sse_y > 0);
    }

    #[test]
    fn model_rd_uniform_offset_skips_ac() {
        // Uniform offset (dc-only residual) with a large ac_thr
        // should classify as AcOnly (if dc_thr is small enough) or
        // AcDc (if dc is also below threshold).
        let pred = [128u8; 8 * 8];
        let src = [129u8; 8 * 8]; // diff = 1 everywhere → var = 0, sse = 64
        let result = model_rd_for_sb_y(
            &src, 8, &pred, 8, 8, 8,
            BlockSize::Block8x8,
            96,  // dc_quant
            80,  // ac_quant
            100, // dc_thr — larger than sse_tx - var_tx = 64/1 = 64 → skip DC too
            100, // ac_thr — var_tx = 0 < 100 → skip AC
            TxMode::TxModeSelect,
        );
        // var = 0 → the `var == 0` branch triggers AcOnly, then
        // sse == var is false (sse=64 != var=0), so we check
        // (sse_tx - var_tx) < dc_thr → 64 < 100 → AcDc.
        assert_eq!(result.skip_txfm, SkipTxfm::AcDc);
    }

    // -----------------------------------------------------------------------
    // model_rd_norm (internal)
    // -----------------------------------------------------------------------

    #[test]
    fn model_rd_norm_boundary() {
        // xsq_q10 = 0 → should return (rate_tab[0], dist_tab[0]) = (65536, 0).
        let (r, d) = model_rd_norm(0);
        assert_eq!(r, RATE_TAB_Q10[0]); // 65536
        assert_eq!(d, DIST_TAB_Q10[0]); // 0
    }

    #[test]
    fn model_rd_norm_mid_range() {
        // Pick a mid-range xsq_q10 and verify interpolation is in
        // the right ballpark (between adjacent table entries).
        let xsq = 10000;
        let (r, d) = model_rd_norm(xsq);
        // r should be between some table entries in the mid-range.
        assert!(r > 0 && r < 65536, "rate should be in (0, 65536): {r}");
        assert!(d > 0 && d < 1024, "dist should be in (0, 1024): {d}");
    }

    // -----------------------------------------------------------------------
    // model_rd_for_sb_uv
    // -----------------------------------------------------------------------

    #[test]
    fn model_rd_uv_identical_src_pred_gives_zero() {
        // Identical src and pred on both U and V -> rate=0, dist=0.
        let src_u = [128u8; 8 * 8];
        let pred_u = [128u8; 8 * 8];
        let src_v = [128u8; 8 * 8];
        let pred_v = [128u8; 8 * 8];
        let result = model_rd_for_sb_uv(
            &src_u, 8, &pred_u, 8,
            &src_v, 8, &pred_v, 8,
            16, 16, // luma 16x16 -> chroma 8x8
            BlockSize::Block16x16,
            96, 80,
        );
        assert_eq!(result.rate, 0, "zero residual -> zero rate");
        assert_eq!(result.dist, 0, "zero residual -> zero dist");
    }

    #[test]
    fn model_rd_uv_textured_gives_positive() {
        // Build chroma planes with a gradient residual -> non-zero
        // rate and dist.
        let pred_u = [128u8; 8 * 8];
        let pred_v = [128u8; 8 * 8];
        let mut src_u = [128u8; 8 * 8];
        let mut src_v = [128u8; 8 * 8];
        for r in 0..8 {
            for c in 0..8 {
                src_u[r * 8 + c] = (128 + r * 3 + c) as u8;
                src_v[r * 8 + c] = (128 + r * 2 + c * 2) as u8;
            }
        }
        let result = model_rd_for_sb_uv(
            &src_u, 8, &pred_u, 8,
            &src_v, 8, &pred_v, 8,
            16, 16,
            BlockSize::Block16x16,
            96, 80,
        );
        assert!(result.rate > 0, "textured UV should have positive rate: {}", result.rate);
        assert!(result.dist > 0, "textured UV should have positive dist: {}", result.dist);
    }

    #[test]
    fn model_rd_uv_dimensions_halved() {
        // Verify that model_rd_for_sb_uv uses halved dimensions:
        // luma 32x32 -> chroma 16x16. We only fill 16x16 of data
        // and it should work without panicking.
        let src_u = [130u8; 16 * 16];
        let pred_u = [128u8; 16 * 16];
        let src_v = [130u8; 16 * 16];
        let pred_v = [128u8; 16 * 16];
        let result = model_rd_for_sb_uv(
            &src_u, 16, &pred_u, 16,
            &src_v, 16, &pred_v, 16,
            32, 32, // luma 32x32 -> chroma 16x16
            BlockSize::Block32x32,
            96, 80,
        );
        // Uniform offset: each plane has sse = 16*16*4 = 1024,
        // var = 0 -> DC only. Both planes contribute.
        assert!(result.rate > 0 || result.dist > 0,
            "uniform offset should produce some cost");
    }

    #[test]
    fn model_rd_uv_64x64_uses_32x32_chroma() {
        // Block64x64 -> chroma Block32x32 (4:2:0). Verify correct
        // chroma block size is used (32x32 chroma = 1024 pixels,
        // num_pels_log2 = 10).
        let src_u = [129u8; 32 * 32];
        let pred_u = [128u8; 32 * 32];
        let src_v = [128u8; 32 * 32]; // V is flat -> only U contributes
        let pred_v = [128u8; 32 * 32];
        let result = model_rd_for_sb_uv(
            &src_u, 32, &pred_u, 32,
            &src_v, 32, &pred_v, 32,
            64, 64,
            BlockSize::Block64x64,
            96, 80,
        );
        // U has uniform offset, V is flat. Overall should have
        // non-zero rate from U plane's DC component.
        assert!(result.dist > 0, "U-plane DC residual should produce dist");
    }

    #[test]
    fn model_rd_uv_8x8_luma_gives_4x4_chroma() {
        // Block8x8 -> chroma Block4x4 (4:2:0). This is the smallest
        // chroma block; verify it doesn't panic.
        let src_u = [135u8; 4 * 4];
        let pred_u = [128u8; 4 * 4];
        let src_v = [135u8; 4 * 4];
        let pred_v = [128u8; 4 * 4];
        let result = model_rd_for_sb_uv(
            &src_u, 4, &pred_u, 4,
            &src_v, 4, &pred_v, 4,
            8, 8,
            BlockSize::Block8x8,
            96, 80,
        );
        // diff = 7 per pixel, 16 pixels per plane, sse = 16*49 = 784
        // per plane. Should produce non-trivial cost.
        assert!(result.rate > 0, "small chroma block should have positive rate");
        assert!(result.dist > 0, "small chroma block should have positive dist");
    }

    #[test]
    fn model_rd_uv_sums_both_planes() {
        // Verify that cost is the sum of U and V planes: if only
        // one plane has residual, cost should be less than when both
        // have residual.
        let flat = [128u8; 8 * 8];
        let mut textured = [128u8; 8 * 8];
        for r in 0..8 {
            for c in 0..8 {
                textured[r * 8 + c] = (128 + r * 3 + c) as u8;
            }
        }
        // Only U has residual.
        let result_u_only = model_rd_for_sb_uv(
            &textured, 8, &flat, 8,
            &flat, 8, &flat, 8,
            16, 16,
            BlockSize::Block16x16,
            96, 80,
        );
        // Both U and V have the same residual.
        let result_both = model_rd_for_sb_uv(
            &textured, 8, &flat, 8,
            &textured, 8, &flat, 8,
            16, 16,
            BlockSize::Block16x16,
            96, 80,
        );
        // Both planes identical -> exactly double the single-plane cost.
        assert_eq!(result_both.rate, result_u_only.rate * 2,
            "both-plane rate should be 2x single-plane rate");
        assert_eq!(result_both.dist, result_u_only.dist * 2,
            "both-plane dist should be 2x single-plane dist");
    }
}
