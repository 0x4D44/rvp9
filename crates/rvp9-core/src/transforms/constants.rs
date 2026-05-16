//! Transform constants shared by forward and inverse DCT/ADST kernels.
//!
//! Sourced verbatim from libvpx `vpx_dsp/txfm_common.h`. The full set of
//! `COSPI_*_64` values (1..32 of 64) and `SINPI_*_9` values (1..4 of 9)
//! are imported as needed by each kernel sub-module. This module lands
//! only the subset used by `idct4` and the 4×4 family; subsequent
//! transform kernel sub-tasks (`idct8`, `idct16`, `idct32`, ADST family)
//! add their constants here.

/// Bits of precision in the DCT/ADST fixed-point constants. All
/// products of an `input` by a `COSPI_*_64` are shifted right by this
/// amount via [`dct_const_round_shift`]. Mirrors libvpx
/// `txfm_common.h:17`.
pub const DCT_CONST_BITS: i32 = 14;

/// `round(cos(8 * pi / 64) * (1 << 14))` = 15137. Mirrors libvpx
/// `txfm_common.h:35`.
pub const COSPI_8_64: i32 = 15137;

/// `round(cos(16 * pi / 64) * (1 << 14))` = 11585 = `sqrt(2)/2 << 14`.
/// Mirrors libvpx `txfm_common.h:43`.
pub const COSPI_16_64: i32 = 11585;

/// `round(cos(24 * pi / 64) * (1 << 14))` = 6270. Mirrors libvpx
/// `txfm_common.h:51`.
pub const COSPI_24_64: i32 = 6270;

/// Symmetric rounding right-shift by `DCT_CONST_BITS`. Mirrors libvpx
/// `dct_const_round_shift` (`inv_txfm.h:38-41`).
#[inline]
#[must_use]
pub const fn dct_const_round_shift(x: i64) -> i64 {
    let rounding = 1_i64 << (DCT_CONST_BITS - 1);
    (x + rounding) >> DCT_CONST_BITS
}

/// `clip_pixel_add(dest, trans)`: residual-add with saturation to
/// `[0, 255]`. Mirrors libvpx `clip_pixel_add` (`inv_txfm.h:117-120`)
/// for the non-HBD path (D-01 RESOLVED: HBD deferred).
///
/// `trans` is the transform output (an `i32`-range signed delta).
#[inline]
#[must_use]
pub fn clip_pixel_add(dest: u8, trans: i64) -> u8 {
    let sum = (dest as i32) + (trans as i32);
    sum.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_const_round_shift_centred() {
        // x = 0 → 0
        assert_eq!(dct_const_round_shift(0), 0);
        // x = (1 << 14) → 1 (with rounding)
        assert_eq!(dct_const_round_shift(1 << 14), 1);
        // x = (1 << 13) - 1 (just below half) → 0
        assert_eq!(dct_const_round_shift((1 << 13) - 1), 0);
        // x = (1 << 13) (exactly half) → 1 (round-half-up)
        assert_eq!(dct_const_round_shift(1 << 13), 1);
        // Negative half: (-(1 << 13)) → 0 because rounding pushes to 0.
        assert_eq!(dct_const_round_shift(-(1 << 13)), 0);
        // -(1 << 14) → -1.
        assert_eq!(dct_const_round_shift(-(1 << 14)), -1);
    }

    #[test]
    fn clip_pixel_add_clamps() {
        assert_eq!(clip_pixel_add(0, 0), 0);
        assert_eq!(clip_pixel_add(128, 10), 138);
        assert_eq!(clip_pixel_add(128, -10), 118);
        assert_eq!(clip_pixel_add(250, 10), 255); // upper clamp
        assert_eq!(clip_pixel_add(5, -10), 0); // lower clamp
        assert_eq!(clip_pixel_add(100, 1_000_000), 255);
        assert_eq!(clip_pixel_add(100, -1_000_000), 0);
    }
}
