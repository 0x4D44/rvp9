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

/// `round(cos(4 * pi / 64) * (1 << 14))` = 16069. Mirrors libvpx
/// `txfm_common.h:31`.
pub const COSPI_4_64: i32 = 16069;

/// `round(cos(12 * pi / 64) * (1 << 14))` = 13623. Mirrors libvpx
/// `txfm_common.h:39`.
pub const COSPI_12_64: i32 = 13623;

/// `round(cos(20 * pi / 64) * (1 << 14))` = 9102. Mirrors libvpx
/// `txfm_common.h:47`.
pub const COSPI_20_64: i32 = 9102;

/// `round(cos(28 * pi / 64) * (1 << 14))` = 3196. Mirrors libvpx
/// `txfm_common.h:55`.
pub const COSPI_28_64: i32 = 3196;

/// `round(cos(2 * pi / 64) * (1 << 14))` = 16305. Mirrors libvpx
/// `txfm_common.h:29`.
pub const COSPI_2_64: i32 = 16305;

/// `round(cos(6 * pi / 64) * (1 << 14))` = 15679. Mirrors libvpx
/// `txfm_common.h:33`.
pub const COSPI_6_64: i32 = 15679;

/// `round(cos(10 * pi / 64) * (1 << 14))` = 14449. Mirrors libvpx
/// `txfm_common.h:37`.
pub const COSPI_10_64: i32 = 14449;

/// `round(cos(14 * pi / 64) * (1 << 14))` = 12665. Mirrors libvpx
/// `txfm_common.h:41`.
pub const COSPI_14_64: i32 = 12665;

/// `round(cos(18 * pi / 64) * (1 << 14))` = 10394. Mirrors libvpx
/// `txfm_common.h:45`.
pub const COSPI_18_64: i32 = 10394;

/// `round(cos(22 * pi / 64) * (1 << 14))` = 7723. Mirrors libvpx
/// `txfm_common.h:49`.
pub const COSPI_22_64: i32 = 7723;

/// `round(cos(26 * pi / 64) * (1 << 14))` = 4756. Mirrors libvpx
/// `txfm_common.h:53`.
pub const COSPI_26_64: i32 = 4756;

/// `round(cos(30 * pi / 64) * (1 << 14))` = 1606. Mirrors libvpx
/// `txfm_common.h:57`.
pub const COSPI_30_64: i32 = 1606;

/// `round(cos(1 * pi / 64) * (1 << 14))` = 16364. Mirrors libvpx
/// `txfm_common.h:28`.
pub const COSPI_1_64: i32 = 16364;

/// `round(cos(3 * pi / 64) * (1 << 14))` = 16207. Mirrors libvpx
/// `txfm_common.h:30`.
pub const COSPI_3_64: i32 = 16207;

/// `round(cos(5 * pi / 64) * (1 << 14))` = 15893. Mirrors libvpx
/// `txfm_common.h:32`.
pub const COSPI_5_64: i32 = 15893;

/// `round(cos(7 * pi / 64) * (1 << 14))` = 15426. Mirrors libvpx
/// `txfm_common.h:34`.
pub const COSPI_7_64: i32 = 15426;

/// `round(cos(9 * pi / 64) * (1 << 14))` = 14811. Mirrors libvpx
/// `txfm_common.h:36`.
pub const COSPI_9_64: i32 = 14811;

/// `round(cos(11 * pi / 64) * (1 << 14))` = 14053. Mirrors libvpx
/// `txfm_common.h:38`.
pub const COSPI_11_64: i32 = 14053;

/// `round(cos(13 * pi / 64) * (1 << 14))` = 13160. Mirrors libvpx
/// `txfm_common.h:40`.
pub const COSPI_13_64: i32 = 13160;

/// `round(cos(15 * pi / 64) * (1 << 14))` = 12140. Mirrors libvpx
/// `txfm_common.h:42`.
pub const COSPI_15_64: i32 = 12140;

/// `round(cos(17 * pi / 64) * (1 << 14))` = 11003. Mirrors libvpx
/// `txfm_common.h:44`.
pub const COSPI_17_64: i32 = 11003;

/// `round(cos(19 * pi / 64) * (1 << 14))` = 9760. Mirrors libvpx
/// `txfm_common.h:46`.
pub const COSPI_19_64: i32 = 9760;

/// `round(cos(21 * pi / 64) * (1 << 14))` = 8423. Mirrors libvpx
/// `txfm_common.h:48`.
pub const COSPI_21_64: i32 = 8423;

/// `round(cos(23 * pi / 64) * (1 << 14))` = 7005. Mirrors libvpx
/// `txfm_common.h:50`.
pub const COSPI_23_64: i32 = 7005;

/// `round(cos(25 * pi / 64) * (1 << 14))` = 5520. Mirrors libvpx
/// `txfm_common.h:52`.
pub const COSPI_25_64: i32 = 5520;

/// `round(cos(27 * pi / 64) * (1 << 14))` = 3981. Mirrors libvpx
/// `txfm_common.h:54`.
pub const COSPI_27_64: i32 = 3981;

/// `round(cos(29 * pi / 64) * (1 << 14))` = 2404. Mirrors libvpx
/// `txfm_common.h:56`.
pub const COSPI_29_64: i32 = 2404;

/// `round(cos(31 * pi / 64) * (1 << 14))` = 804. Mirrors libvpx
/// `txfm_common.h:58`.
pub const COSPI_31_64: i32 = 804;

/// Symmetric rounding right-shift by `DCT_CONST_BITS`. Mirrors libvpx
/// `dct_const_round_shift` (`inv_txfm.h:38-41`).
#[inline]
#[must_use]
pub const fn dct_const_round_shift(x: i64) -> i64 {
    let rounding = 1_i64 << (DCT_CONST_BITS - 1);
    (x + rounding) >> DCT_CONST_BITS
}

/// `WRAPLOW(x)` — modular truncation to 16 bits (sign-extended back to
/// `i32`). Mirrors libvpx `WRAPLOW` (`inv_txfm.h:87`) **plus the implicit
/// `int16_t step[]` storage truncation** that happens on every store of
/// a `dct_const_round_shift` result in the C kernels.
///
/// In default libvpx builds (`CONFIG_EMULATE_HARDWARE=0`) the textual
/// `WRAPLOW(x)` macro is itself a no-op `(int32_t)x` cast — but the
/// step arrays are declared `int16_t`, so each `step[i] = WRAPLOW(...)`
/// assignment implicitly truncates `int32 → int16` (C 6.3.1.3, two's
/// complement modular reduction on every conforming compiler). We
/// reproduce that behaviour explicitly via `as i16 as i32`.
///
/// For typical spec-conformant coefficient magnitudes (post
/// `dct_const_round_shift`, values fit in 16 bits), this is a no-op.
/// For adversarial or out-of-range inputs it preserves bit-exactness
/// against libvpx.
#[inline]
#[must_use]
pub const fn wraplow(x: i32) -> i32 {
    x as i16 as i32
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
