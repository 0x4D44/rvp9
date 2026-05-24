//! VP9 quantizer lookup tables and `qindex → qstep` mapping.
//!
//! Owns the six DC/AC quantization LUTs (8-bit + 10-bit + 12-bit) and
//! the bit-clamped lookups [`dc_quant`] / [`ac_quant`]. Mirrors libvpx
//! `vp9_quant_common.c` including the `vpx_bit_depth_t` dispatch
//! (`vp9_dc_quant` / `vp9_ac_quant` at `vp9_quant_common.c:164-194`).
//!
//! Encoder-only conversion functions (`vp9_quantizer_to_qindex`,
//! `vp9_qindex_to_quantizer`) live in `rvp9-encoder::quantize` per
//! quant LLD §7.5; this module is the bit-exact lookup surface used
//! by both the decoder (dequant) and the encoder (quant-table init).
//!
//! ## Status
//!
//! Stage 3 task 4a lands the 8-bit LUTs + the four conversion functions
//! (`dc_quant`, `ac_quant`, `apply_qindex_delta`, `get_qindex`). Track T
//! phase 0 (T-1 HBD scaffolding) adds the four HBD LUTs and threads a
//! [`BitDepth`] parameter through `dc_quant` / `ac_quant`; today every
//! caller passes `BitDepth::Eight` so behaviour is unchanged.

pub mod forward;
pub mod tables;

pub use forward::{
    PlaneQuantizers, QuantizerDeltas, init_plane_quantizers, plane_dequant_pair, vp9_quantize_b,
    vp9_quantize_b_32x32, vp9_quantize_fp, vp9_quantize_fp_32x32,
};
pub use tables::{
    AC_QLOOKUP, AC_QLOOKUP_10, AC_QLOOKUP_12, DC_QLOOKUP, DC_QLOOKUP_10, DC_QLOOKUP_12,
};

use crate::bitstream::BitDepth;

/// Maximum value of a clamped qindex. `MAXQ` in libvpx
/// (`vp9_quant_common.h:22`).
pub const MAXQ: i32 = 255;

/// Minimum value of a clamped qindex. `MINQ` in libvpx (== 0).
pub const MINQ: i32 = 0;

/// Number of valid qindex values. `QINDEX_RANGE` in libvpx (== 256).
pub const QINDEX_RANGE: usize = (MAXQ - MINQ + 1) as usize;

/// Return the 16-bit DC quantizer for `(qindex + delta)` clamped into
/// `[MINQ, MAXQ]`. Mirrors `vp9_dc_quant` (`vp9_quant_common.c:164-178`)
/// — selects between [`DC_QLOOKUP`] / [`DC_QLOOKUP_10`] / [`DC_QLOOKUP_12`]
/// based on `bit_depth`.
///
/// `qindex` and `delta` are passed as `i32` to faithfully mirror the
/// libvpx C signature; the implementation clamps to a valid LUT index.
#[inline]
#[must_use]
pub fn dc_quant(qindex: i32, delta: i32, bit_depth: BitDepth) -> i16 {
    let idx = apply_qindex_delta(qindex, delta) as usize;
    match bit_depth {
        BitDepth::Eight => DC_QLOOKUP[idx],
        BitDepth::Ten => DC_QLOOKUP_10[idx],
        BitDepth::Twelve => DC_QLOOKUP_12[idx],
    }
}

/// Return the 16-bit AC quantizer for `(qindex + delta)` clamped into
/// `[MINQ, MAXQ]`. Mirrors `vp9_ac_quant` (`vp9_quant_common.c:180-194`)
/// — selects between [`AC_QLOOKUP`] / [`AC_QLOOKUP_10`] / [`AC_QLOOKUP_12`]
/// based on `bit_depth`.
#[inline]
#[must_use]
pub fn ac_quant(qindex: i32, delta: i32, bit_depth: BitDepth) -> i16 {
    let idx = apply_qindex_delta(qindex, delta) as usize;
    match bit_depth {
        BitDepth::Eight => AC_QLOOKUP[idx],
        BitDepth::Ten => AC_QLOOKUP_10[idx],
        BitDepth::Twelve => AC_QLOOKUP_12[idx],
    }
}

/// Sum a qindex with a signed delta and clamp into `[MINQ, MAXQ]`.
/// Equivalent to libvpx's `clamp(qindex + delta, 0, MAXQ)` idiom.
#[inline]
#[must_use]
pub fn apply_qindex_delta(qindex: i32, delta: i32) -> i32 {
    qindex.saturating_add(delta).clamp(MINQ, MAXQ)
}

/// Scalar reference for the Wave H H6 `plane_dequant_finisher` dispatch
/// slot — pass 2 of the split detokenize loop in
/// [`crate::entropy::detokenize::decode_coefs`].
///
/// Replays the per-coefficient dequant arithmetic the bool-coder loop
/// deposits into four parallel scratch arrays (`raster`, `dqv`, `val`,
/// `sign`). For each scratch entry `i`:
///
/// ```text
/// mag = (val[i] * dqv[i]) >> dq_shift
/// dq_coefs_out[raster[i]] = if sign[i] != 0 { -mag } else { mag }
/// ```
///
/// `dq_shift` is `0` for every transform size except `Tx32x32` (which
/// uses `1`). The output buffer is assumed pre-zeroed (the detokenize
/// caller zeros `dq_coefs_out[..seg_eob]` at function entry).
///
/// This is the AVX2-kernel sibling for the slot
/// [`crate::dsp::SimdDispatch::plane_dequant_finisher`]; the AVX2 path
/// processes 8 i32 lanes per iteration and a scalar tail.
pub fn plane_dequant_finisher(
    scratch_raster: &[u16],
    scratch_dqv: &[i16],
    scratch_val: &[i16],
    scratch_sign: &[u8],
    dq_shift: u32,
    dq_coefs_out: &mut [i32],
) {
    debug_assert_eq!(scratch_raster.len(), scratch_dqv.len());
    debug_assert_eq!(scratch_raster.len(), scratch_val.len());
    debug_assert_eq!(scratch_raster.len(), scratch_sign.len());
    for i in 0..scratch_raster.len() {
        let raster_idx = scratch_raster[i] as usize;
        let dqv_i = i32::from(scratch_dqv[i]);
        let val_i = i32::from(scratch_val[i]);
        let sign = scratch_sign[i];
        let mag = (val_i * dqv_i) >> dq_shift;
        dq_coefs_out[raster_idx] = if sign != 0 { -mag } else { mag };
    }
}

/// Compute the effective qindex for a (segment_alt_q, base_qindex) pair.
/// Mirrors `vp9_get_qindex` (`vp9_quant_common.c:196-206`).
///
/// `seg_alt_q` is the segment-feature value (signed `i32`); `abs_delta`
/// is `true` for `SEGMENT_ABSDATA` (treat the segment value as an
/// absolute qindex) and `false` for delta-mode (add to base_qindex).
///
/// Callers gate this on whether the SEG_LVL_ALT_Q feature is active
/// for the segment (the `segfeature_active` check at
/// `vp9_quant_common.c:198`); this function assumes activation.
#[inline]
#[must_use]
pub fn get_qindex(seg_alt_q: i32, abs_delta: bool, base_qindex: i32) -> i32 {
    let raw = if abs_delta {
        seg_alt_q
    } else {
        base_qindex.saturating_add(seg_alt_q)
    };
    raw.clamp(MINQ, MAXQ)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Boundary values of dc_quant — first entry is 4 (vp9_quant_common.c:16),
    /// last entry is 1336 (vp9_quant_common.c:34).
    #[test]
    fn dc_quant_boundary() {
        assert_eq!(dc_quant(0, 0, BitDepth::Eight), 4);
        assert_eq!(dc_quant(MAXQ, 0, BitDepth::Eight), 1336);
    }

    /// Boundary values of ac_quant — first entry is 4, last is 1828.
    #[test]
    fn ac_quant_boundary() {
        assert_eq!(ac_quant(0, 0, BitDepth::Eight), 4);
        assert_eq!(ac_quant(MAXQ, 0, BitDepth::Eight), 1828);
    }

    /// Negative deltas clamp at MINQ.
    #[test]
    fn delta_clamp_below() {
        assert_eq!(apply_qindex_delta(0, -10), 0);
        assert_eq!(apply_qindex_delta(5, -100), 0);
        assert_eq!(dc_quant(0, -10, BitDepth::Eight), DC_QLOOKUP[0]);
    }

    /// Positive deltas clamp at MAXQ.
    #[test]
    fn delta_clamp_above() {
        assert_eq!(apply_qindex_delta(MAXQ, 10), MAXQ);
        assert_eq!(apply_qindex_delta(250, 100), MAXQ);
        assert_eq!(
            dc_quant(MAXQ, 10, BitDepth::Eight),
            DC_QLOOKUP[MAXQ as usize]
        );
    }

    /// Spot checks at index 1 (unambiguous near the table head — second
    /// entry of each LUT is `8`).
    #[test]
    fn spot_check_index_1() {
        assert_eq!(dc_quant(1, 0, BitDepth::Eight), 8);
        assert_eq!(ac_quant(1, 0, BitDepth::Eight), 8);
    }

    /// Mid-range spot check verified by hand-counting libvpx
    /// `vp9_quant_common.c` row layouts:
    /// - DC row 5 (indices 70-83, 14 entries): 66, 66, 67, 68, 69, 70,
    ///   70, 71, 72, 73, 74, 74, 75, 76 → `dc[80] = 74`
    /// - AC row 5 (indices 65-77, 13 entries): 72..=84, then row 6
    ///   (78-90) is 85..=97 → `ac[80] = 87`
    #[test]
    fn mid_range_spot_check() {
        assert_eq!(dc_quant(80, 0, BitDepth::Eight), 74);
        assert_eq!(ac_quant(80, 0, BitDepth::Eight), 87);
    }

    /// HBD dispatch — Track T phase 0 (T-1 HBD scaffolding). Boundary
    /// values from libvpx `vp9_quant_common.c` for the 10/12-bit LUTs.
    #[test]
    fn hbd_dispatch_boundary() {
        // 10-bit endpoints (vp9_quant_common.c:38-59 / 113-134).
        assert_eq!(dc_quant(0, 0, BitDepth::Ten), 4);
        assert_eq!(dc_quant(MAXQ, 0, BitDepth::Ten), 5347);
        assert_eq!(ac_quant(0, 0, BitDepth::Ten), 4);
        assert_eq!(ac_quant(MAXQ, 0, BitDepth::Ten), 7312);
        // 12-bit endpoints (vp9_quant_common.c:61-86 / 136-161).
        assert_eq!(dc_quant(0, 0, BitDepth::Twelve), 4);
        assert_eq!(dc_quant(MAXQ, 0, BitDepth::Twelve), 21387);
        assert_eq!(ac_quant(0, 0, BitDepth::Twelve), 4);
        assert_eq!(ac_quant(MAXQ, 0, BitDepth::Twelve), 29247);
    }

    /// HBD second-entry spot check (qindex = 1).
    /// libvpx values: `dc_qlookup_10[1] = 9`, `ac_qlookup_10[1] = 9`,
    /// `dc_qlookup_12[1] = 12`, `ac_qlookup_12[1] = 13`.
    #[test]
    fn hbd_dispatch_spot_check_index_1() {
        assert_eq!(dc_quant(1, 0, BitDepth::Ten), 9);
        assert_eq!(ac_quant(1, 0, BitDepth::Ten), 9);
        assert_eq!(dc_quant(1, 0, BitDepth::Twelve), 12);
        assert_eq!(ac_quant(1, 0, BitDepth::Twelve), 13);
    }

    /// `get_qindex` in delta mode: base + seg, clamped.
    #[test]
    fn get_qindex_delta_mode() {
        assert_eq!(get_qindex(5, false, 100), 105);
        assert_eq!(get_qindex(-10, false, 5), 0); // clamp
        assert_eq!(get_qindex(100, false, 200), MAXQ); // clamp
    }

    /// `get_qindex` in absdata mode: seg_alt_q directly.
    #[test]
    fn get_qindex_abs_mode() {
        assert_eq!(get_qindex(150, true, 100), 150);
        assert_eq!(get_qindex(-5, true, 100), 0); // clamp
        assert_eq!(get_qindex(300, true, 100), MAXQ); // clamp
    }

    /// All 256 entries of both LUTs are non-negative and monotonically
    /// non-decreasing (libvpx invariant — qstep is a monotone fn of
    /// qindex).
    #[test]
    fn luts_are_monotonic_nondecreasing() {
        for i in 1..QINDEX_RANGE {
            assert!(
                DC_QLOOKUP[i] >= DC_QLOOKUP[i - 1],
                "DC_QLOOKUP not monotonic at index {i}: {} < {}",
                DC_QLOOKUP[i],
                DC_QLOOKUP[i - 1]
            );
            assert!(
                AC_QLOOKUP[i] >= AC_QLOOKUP[i - 1],
                "AC_QLOOKUP not monotonic at index {i}: {} < {}",
                AC_QLOOKUP[i],
                AC_QLOOKUP[i - 1]
            );
        }
    }
}
