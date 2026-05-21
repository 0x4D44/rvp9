// Wave A G11 — workspace lint `clippy::print_stderr` is "deny" for
// library crates. The `m13_v_block_24_12_iht4x4_probe` test dumps
// intermediate iht4x4 output to stderr while diagnosing the M13
// V-block divergence. Scheduled for triage in the HLD §G11 deliverable
// 3 eprintln audit follow-up. Allow at the module level until that
// sweep lands.
#![allow(clippy::print_stderr)]

//! Inverse transform scalar reference kernels (decoder + encoder
//! reconstruction).
//!
//! Currently lands the 4×4, 8×8 and 16×16 inverse-DCT family:
//!
//! - [`idct4`] / [`idct4x4_16_add`] / [`idct4x4_1_add`] — mirrors
//!   libvpx `idct4_c`, `vpx_idct4x4_16_add_c`, `vpx_idct4x4_1_add_c`
//!   (`vpx_dsp/inv_txfm.c:133-195`).
//! - [`idct8`] / [`idct8x8_64_add`] / [`idct8x8_1_add`] — mirrors
//!   `idct8_c`, `vpx_idct8x8_64_add_c`, `vpx_idct8x8_1_add_c`
//!   (`vpx_dsp/inv_txfm.c:271-387`).
//! - [`idct16`] / [`idct16x16_256_add`] / [`idct16x16_1_add`] — mirrors
//!   `idct16_c`, `vpx_idct16x16_256_add_c`, `vpx_idct16x16_1_add_c`
//!   (`vpx_dsp/inv_txfm.c:557-811`).
//! - [`idct32`] / [`idct32x32_1024_add`] / [`idct32x32_1_add`] — mirrors
//!   `idct32_c`, `vpx_idct32x32_1024_add_c`, `vpx_idct32x32_1_add_c`
//!   (`vpx_dsp/inv_txfm.c:813-1276`).
//! - [`iadst4`] / [`iadst8`] / [`iadst16`] — 1D inverse asymmetric
//!   discrete sine transforms. Mirror libvpx `iadst4_c`, `iadst8_c`,
//!   `iadst16_c` (`vpx_dsp/inv_txfm.c:96, 196, 389`).
//! - [`iht4x4_16_add`] / [`iht8x8_64_add`] / [`iht16x16_256_add`] —
//!   hybrid 2D inverse transforms that dispatch on
//!   [`super::TxType`] (`DCT_DCT` / `ADST_DCT` / `DCT_ADST` /
//!   `ADST_ADST`). Mirror libvpx `vp9_iht4x4_16_add_c`,
//!   `vp9_iht8x8_64_add_c`, `vp9_iht16x16_256_add_c`
//!   (`vp9/common/vp9_idct.c:20, 59, 92`).
//!
//! All kernels apply [`wraplow`] (i16 modular truncation) at every site
//! where the C source assigns a `dct_const_round_shift` result into an
//! `int16_t step[]` slot. This preserves bit-exactness against libvpx
//! for inputs whose intermediates exceed 16 bits.

use super::types::TxType;
use super::constants::{
    COSPI_1_64, COSPI_2_64, COSPI_3_64, COSPI_4_64, COSPI_5_64, COSPI_6_64, COSPI_7_64, COSPI_8_64,
    COSPI_9_64, COSPI_10_64, COSPI_11_64, COSPI_12_64, COSPI_13_64, COSPI_14_64, COSPI_15_64,
    COSPI_16_64, COSPI_17_64, COSPI_18_64, COSPI_19_64, COSPI_20_64, COSPI_21_64, COSPI_22_64,
    COSPI_23_64, COSPI_24_64, COSPI_25_64, COSPI_26_64, COSPI_27_64, COSPI_28_64, COSPI_29_64,
    COSPI_30_64, COSPI_31_64, SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9, clip_pixel_add,
    dct_const_round_shift, wraplow,
};

/// 1D 4-point inverse DCT. Output values are stored as `i32` but each
/// step and output assignment is `wraplow`-truncated to mirror the
/// implicit `int16_t step[]` storage in libvpx.
///
/// Mirrors libvpx `idct4_c` (`vpx_dsp/inv_txfm.c:133-152`).
#[inline]
pub fn idct4(input: &[i32; 4], output: &mut [i32; 4]) {
    // stage 1 — libvpx stores into `int16_t step[4]`; we mirror with `wraplow`.
    let temp1: i64 = (input[0] as i64 + input[2] as i64) * COSPI_16_64 as i64;
    let temp2: i64 = (input[0] as i64 - input[2] as i64) * COSPI_16_64 as i64;
    let step0: i32 = wraplow(dct_const_round_shift(temp1) as i32);
    let step1: i32 = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1: i64 =
        input[1] as i64 * COSPI_24_64 as i64 - input[3] as i64 * COSPI_8_64 as i64;
    let temp2: i64 =
        input[1] as i64 * COSPI_8_64 as i64 + input[3] as i64 * COSPI_24_64 as i64;
    let step2: i32 = wraplow(dct_const_round_shift(temp1) as i32);
    let step3: i32 = wraplow(dct_const_round_shift(temp2) as i32);

    // stage 2 — `output` is `tran_low_t` (`int16_t` in default libvpx
    // build), so each store also truncates.
    output[0] = wraplow(step0 + step3);
    output[1] = wraplow(step1 + step2);
    output[2] = wraplow(step1 - step2);
    output[3] = wraplow(step0 - step3);
}

/// 2D 16-coefficient inverse 4×4 DCT with residual add to `dest`.
///
/// `input` is the 16 dequantised coefficients in raster order.
/// `dest[r * stride + c]` is the destination pixel; the transform
/// output is added to `dest` and saturated to `[0, 255]`.
///
/// Mirrors `vpx_idct4x4_16_add_c` (`vpx_dsp/inv_txfm.c:154-176`).
pub fn idct4x4_16_add(input: &[i32; 16], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 3 + 4, "dest too small for 4x4 add");

    let mut intermediate = [0i32; 16];

    // Row pass.
    for r in 0..4 {
        let row_in: &[i32; 4] = input[r * 4..r * 4 + 4].try_into().unwrap();
        let row_out_mut: &mut [i32; 4] = (&mut intermediate[r * 4..r * 4 + 4])
            .try_into()
            .unwrap();
        idct4(row_in, row_out_mut);
    }

    // Column pass + residual add.
    for c in 0..4 {
        let col_in: [i32; 4] = [
            intermediate[c],
            intermediate[4 + c],
            intermediate[8 + c],
            intermediate[12 + c],
        ];
        let mut col_out = [0i32; 4];
        idct4(&col_in, &mut col_out);
        for r in 0..4 {
            // ROUND_POWER_OF_TWO(col_out[r], 4)  per libvpx inv_txfm.c:173
            let shifted = (col_out[r] as i64 + 8) >> 4;
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], shifted);
        }
    }
}

/// 2D 1-coefficient inverse 4×4 DCT (DC-only fast path).
///
/// `input` is a single coefficient at position 0 (DC); the remaining
/// 15 coefficients are assumed zero. Mirrors `vpx_idct4x4_1_add_c`
/// (`vpx_dsp/inv_txfm.c:178-195`).
pub fn idct4x4_1_add(input: i32, dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 3 + 4, "dest too small for 4x4 add");

    // Two passes of (input * COSPI_16_64) with `wraplow` truncation to
    // mirror libvpx `tran_low_t out` (int16_t in default build), then
    // a >>4 final shift.
    let out1 = wraplow(dct_const_round_shift((input as i64) * (COSPI_16_64 as i64)) as i32);
    let out2 = wraplow(dct_const_round_shift((out1 as i64) * (COSPI_16_64 as i64)) as i32);
    let a1 = ((out2 as i64) + 8) >> 4;
    for r in 0..4 {
        for c in 0..4 {
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], a1);
        }
    }
}

/// 1D 8-point inverse DCT.
///
/// Mirrors libvpx `idct8_c` (`vpx_dsp/inv_txfm.c:271-324`). Every
/// `step1[]` / `step2[]` store applies [`wraplow`] to preserve libvpx's
/// implicit `int16_t` storage truncation.
#[inline]
pub fn idct8(input: &[i32; 8], output: &mut [i32; 8]) {
    let mut step1 = [0i32; 8];
    let mut step2 = [0i32; 8];

    // stage 1 — store inputs directly; values are `tran_low_t`
    // (already int16 in default libvpx).
    step1[0] = input[0];
    step1[2] = input[4];
    step1[1] = input[2];
    step1[3] = input[6];
    let temp1 = input[1] as i64 * COSPI_28_64 as i64 - input[7] as i64 * COSPI_4_64 as i64;
    let temp2 = input[1] as i64 * COSPI_4_64 as i64 + input[7] as i64 * COSPI_28_64 as i64;
    step1[4] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[7] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = input[5] as i64 * COSPI_12_64 as i64 - input[3] as i64 * COSPI_20_64 as i64;
    let temp2 = input[5] as i64 * COSPI_20_64 as i64 + input[3] as i64 * COSPI_12_64 as i64;
    step1[5] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[6] = wraplow(dct_const_round_shift(temp2) as i32);

    // stage 2
    let temp1 = (step1[0] as i64 + step1[2] as i64) * COSPI_16_64 as i64;
    let temp2 = (step1[0] as i64 - step1[2] as i64) * COSPI_16_64 as i64;
    step2[0] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[1] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = step1[1] as i64 * COSPI_24_64 as i64 - step1[3] as i64 * COSPI_8_64 as i64;
    let temp2 = step1[1] as i64 * COSPI_8_64 as i64 + step1[3] as i64 * COSPI_24_64 as i64;
    step2[2] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[3] = wraplow(dct_const_round_shift(temp2) as i32);
    step2[4] = wraplow(step1[4] + step1[5]);
    step2[5] = wraplow(step1[4] - step1[5]);
    step2[6] = wraplow(-step1[6] + step1[7]);
    step2[7] = wraplow(step1[6] + step1[7]);

    // stage 3
    step1[0] = wraplow(step2[0] + step2[3]);
    step1[1] = wraplow(step2[1] + step2[2]);
    step1[2] = wraplow(step2[1] - step2[2]);
    step1[3] = wraplow(step2[0] - step2[3]);
    step1[4] = step2[4];
    let temp1 = (step2[6] as i64 - step2[5] as i64) * COSPI_16_64 as i64;
    let temp2 = (step2[5] as i64 + step2[6] as i64) * COSPI_16_64 as i64;
    step1[5] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[6] = wraplow(dct_const_round_shift(temp2) as i32);
    step1[7] = step2[7];

    // stage 4 — output is `tran_low_t`, truncate on store.
    output[0] = wraplow(step1[0] + step1[7]);
    output[1] = wraplow(step1[1] + step1[6]);
    output[2] = wraplow(step1[2] + step1[5]);
    output[3] = wraplow(step1[3] + step1[4]);
    output[4] = wraplow(step1[3] - step1[4]);
    output[5] = wraplow(step1[2] - step1[5]);
    output[6] = wraplow(step1[1] - step1[6]);
    output[7] = wraplow(step1[0] - step1[7]);
}

/// 2D 64-coefficient inverse 8×8 DCT with residual add to `dest`.
///
/// `input[64]` is the dequantised coefficients in raster order;
/// `dest[r * stride + c]` is the destination plane. Output is added
/// to `dest` and saturated to `[0, 255]`.
///
/// Mirrors `vpx_idct8x8_64_add_c` (`vpx_dsp/inv_txfm.c:326-348`).
pub fn idct8x8_64_add(input: &[i32; 64], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 7 + 8, "dest too small for 8x8 add");

    let mut intermediate = [0i32; 64];

    // Row pass.
    for r in 0..8 {
        let row_in: &[i32; 8] = input[r * 8..r * 8 + 8].try_into().unwrap();
        let row_out_mut: &mut [i32; 8] = (&mut intermediate[r * 8..r * 8 + 8])
            .try_into()
            .unwrap();
        idct8(row_in, row_out_mut);
    }

    // Column pass + residual add (ROUND_POWER_OF_TWO with shift 5).
    for c in 0..8 {
        let col_in: [i32; 8] = [
            intermediate[c],
            intermediate[8 + c],
            intermediate[16 + c],
            intermediate[24 + c],
            intermediate[32 + c],
            intermediate[40 + c],
            intermediate[48 + c],
            intermediate[56 + c],
        ];
        let mut col_out = [0i32; 8];
        idct8(&col_in, &mut col_out);
        for r in 0..8 {
            let shifted = (col_out[r] as i64 + 16) >> 5;
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], shifted);
        }
    }
}

/// 2D 1-coefficient inverse 8×8 DCT (DC-only fast path).
///
/// `input` is `coeffs[0]`; the remaining 63 coefficients are assumed
/// zero. Mirrors `vpx_idct8x8_1_add_c` (`vpx_dsp/inv_txfm.c:375-387`).
pub fn idct8x8_1_add(input: i32, dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 7 + 8, "dest too small for 8x8 add");

    let out1 = wraplow(dct_const_round_shift((input as i64) * (COSPI_16_64 as i64)) as i32);
    let out2 = wraplow(dct_const_round_shift((out1 as i64) * (COSPI_16_64 as i64)) as i32);
    let a1 = ((out2 as i64) + 16) >> 5;
    for r in 0..8 {
        for c in 0..8 {
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], a1);
        }
    }
}

/// 1D 16-point inverse DCT.
///
/// Mirrors libvpx `idct16_c` (`vpx_dsp/inv_txfm.c:557-720`). Seven-stage
/// butterfly using cospi at frequencies 2/4/6/8/10/12/14/16/18/20/22/
/// 24/26/28/30. Every `step1[]`/`step2[]` store applies [`wraplow`].
#[inline]
pub fn idct16(input: &[i32; 16], output: &mut [i32; 16]) {
    let mut step1 = [0i32; 16];
    let mut step2 = [0i32; 16];

    // stage 1 — bit-reversed input load (mirrors libvpx :562-577).
    step1[0] = input[0];
    step1[1] = input[8];
    step1[2] = input[4];
    step1[3] = input[12];
    step1[4] = input[2];
    step1[5] = input[10];
    step1[6] = input[6];
    step1[7] = input[14];
    step1[8] = input[1];
    step1[9] = input[9];
    step1[10] = input[5];
    step1[11] = input[13];
    step1[12] = input[3];
    step1[13] = input[11];
    step1[14] = input[7];
    step1[15] = input[15];

    // stage 2 — high-frequency butterflies.
    step2[0] = step1[0];
    step2[1] = step1[1];
    step2[2] = step1[2];
    step2[3] = step1[3];
    step2[4] = step1[4];
    step2[5] = step1[5];
    step2[6] = step1[6];
    step2[7] = step1[7];

    let temp1 = step1[8] as i64 * COSPI_30_64 as i64 - step1[15] as i64 * COSPI_2_64 as i64;
    let temp2 = step1[8] as i64 * COSPI_2_64 as i64 + step1[15] as i64 * COSPI_30_64 as i64;
    step2[8] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[15] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step1[9] as i64 * COSPI_14_64 as i64 - step1[14] as i64 * COSPI_18_64 as i64;
    let temp2 = step1[9] as i64 * COSPI_18_64 as i64 + step1[14] as i64 * COSPI_14_64 as i64;
    step2[9] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[14] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step1[10] as i64 * COSPI_22_64 as i64 - step1[13] as i64 * COSPI_10_64 as i64;
    let temp2 = step1[10] as i64 * COSPI_10_64 as i64 + step1[13] as i64 * COSPI_22_64 as i64;
    step2[10] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[13] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step1[11] as i64 * COSPI_6_64 as i64 - step1[12] as i64 * COSPI_26_64 as i64;
    let temp2 = step1[11] as i64 * COSPI_26_64 as i64 + step1[12] as i64 * COSPI_6_64 as i64;
    step2[11] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[12] = wraplow(dct_const_round_shift(temp2) as i32);

    // stage 3
    step1[0] = step2[0];
    step1[1] = step2[1];
    step1[2] = step2[2];
    step1[3] = step2[3];

    let temp1 = step2[4] as i64 * COSPI_28_64 as i64 - step2[7] as i64 * COSPI_4_64 as i64;
    let temp2 = step2[4] as i64 * COSPI_4_64 as i64 + step2[7] as i64 * COSPI_28_64 as i64;
    step1[4] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[7] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = step2[5] as i64 * COSPI_12_64 as i64 - step2[6] as i64 * COSPI_20_64 as i64;
    let temp2 = step2[5] as i64 * COSPI_20_64 as i64 + step2[6] as i64 * COSPI_12_64 as i64;
    step1[5] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[6] = wraplow(dct_const_round_shift(temp2) as i32);

    step1[8] = wraplow(step2[8] + step2[9]);
    step1[9] = wraplow(step2[8] - step2[9]);
    step1[10] = wraplow(-step2[10] + step2[11]);
    step1[11] = wraplow(step2[10] + step2[11]);
    step1[12] = wraplow(step2[12] + step2[13]);
    step1[13] = wraplow(step2[12] - step2[13]);
    step1[14] = wraplow(-step2[14] + step2[15]);
    step1[15] = wraplow(step2[14] + step2[15]);

    // stage 4
    let temp1 = (step1[0] as i64 + step1[1] as i64) * COSPI_16_64 as i64;
    let temp2 = (step1[0] as i64 - step1[1] as i64) * COSPI_16_64 as i64;
    step2[0] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[1] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = step1[2] as i64 * COSPI_24_64 as i64 - step1[3] as i64 * COSPI_8_64 as i64;
    let temp2 = step1[2] as i64 * COSPI_8_64 as i64 + step1[3] as i64 * COSPI_24_64 as i64;
    step2[2] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[3] = wraplow(dct_const_round_shift(temp2) as i32);
    step2[4] = wraplow(step1[4] + step1[5]);
    step2[5] = wraplow(step1[4] - step1[5]);
    step2[6] = wraplow(-step1[6] + step1[7]);
    step2[7] = wraplow(step1[6] + step1[7]);

    step2[8] = step1[8];
    step2[15] = step1[15];
    let temp1 = -(step1[9] as i64) * COSPI_8_64 as i64 + step1[14] as i64 * COSPI_24_64 as i64;
    let temp2 = step1[9] as i64 * COSPI_24_64 as i64 + step1[14] as i64 * COSPI_8_64 as i64;
    step2[9] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[14] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = -(step1[10] as i64) * COSPI_24_64 as i64 - step1[13] as i64 * COSPI_8_64 as i64;
    let temp2 = -(step1[10] as i64) * COSPI_8_64 as i64 + step1[13] as i64 * COSPI_24_64 as i64;
    step2[10] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[13] = wraplow(dct_const_round_shift(temp2) as i32);
    step2[11] = step1[11];
    step2[12] = step1[12];

    // stage 5
    step1[0] = wraplow(step2[0] + step2[3]);
    step1[1] = wraplow(step2[1] + step2[2]);
    step1[2] = wraplow(step2[1] - step2[2]);
    step1[3] = wraplow(step2[0] - step2[3]);
    step1[4] = step2[4];
    let temp1 = (step2[6] as i64 - step2[5] as i64) * COSPI_16_64 as i64;
    let temp2 = (step2[5] as i64 + step2[6] as i64) * COSPI_16_64 as i64;
    step1[5] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[6] = wraplow(dct_const_round_shift(temp2) as i32);
    step1[7] = step2[7];

    step1[8] = wraplow(step2[8] + step2[11]);
    step1[9] = wraplow(step2[9] + step2[10]);
    step1[10] = wraplow(step2[9] - step2[10]);
    step1[11] = wraplow(step2[8] - step2[11]);
    step1[12] = wraplow(-step2[12] + step2[15]);
    step1[13] = wraplow(-step2[13] + step2[14]);
    step1[14] = wraplow(step2[13] + step2[14]);
    step1[15] = wraplow(step2[12] + step2[15]);

    // stage 6
    step2[0] = wraplow(step1[0] + step1[7]);
    step2[1] = wraplow(step1[1] + step1[6]);
    step2[2] = wraplow(step1[2] + step1[5]);
    step2[3] = wraplow(step1[3] + step1[4]);
    step2[4] = wraplow(step1[3] - step1[4]);
    step2[5] = wraplow(step1[2] - step1[5]);
    step2[6] = wraplow(step1[1] - step1[6]);
    step2[7] = wraplow(step1[0] - step1[7]);
    step2[8] = step1[8];
    step2[9] = step1[9];
    let temp1 = (-(step1[10] as i64) + step1[13] as i64) * COSPI_16_64 as i64;
    let temp2 = (step1[10] as i64 + step1[13] as i64) * COSPI_16_64 as i64;
    step2[10] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[13] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = (-(step1[11] as i64) + step1[12] as i64) * COSPI_16_64 as i64;
    let temp2 = (step1[11] as i64 + step1[12] as i64) * COSPI_16_64 as i64;
    step2[11] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[12] = wraplow(dct_const_round_shift(temp2) as i32);
    step2[14] = step1[14];
    step2[15] = step1[15];

    // stage 7 — output is `tran_low_t`, truncate on store.
    output[0] = wraplow(step2[0] + step2[15]);
    output[1] = wraplow(step2[1] + step2[14]);
    output[2] = wraplow(step2[2] + step2[13]);
    output[3] = wraplow(step2[3] + step2[12]);
    output[4] = wraplow(step2[4] + step2[11]);
    output[5] = wraplow(step2[5] + step2[10]);
    output[6] = wraplow(step2[6] + step2[9]);
    output[7] = wraplow(step2[7] + step2[8]);
    output[8] = wraplow(step2[7] - step2[8]);
    output[9] = wraplow(step2[6] - step2[9]);
    output[10] = wraplow(step2[5] - step2[10]);
    output[11] = wraplow(step2[4] - step2[11]);
    output[12] = wraplow(step2[3] - step2[12]);
    output[13] = wraplow(step2[2] - step2[13]);
    output[14] = wraplow(step2[1] - step2[14]);
    output[15] = wraplow(step2[0] - step2[15]);
}

/// 2D 256-coefficient inverse 16×16 DCT with residual add to `dest`.
///
/// `input[256]` is the dequantised coefficients in raster order;
/// `dest[r * stride + c]` is the destination plane. Output is added to
/// `dest` and saturated to `[0, 255]`. The column-pass uses
/// `ROUND_POWER_OF_TWO(_, 6)` per `vpx_idct16x16_256_add_c`
/// (`vpx_dsp/inv_txfm.c:722-745`).
pub fn idct16x16_256_add(input: &[i32; 256], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 15 + 16, "dest too small for 16x16 add");

    let mut intermediate = [0i32; 256];

    // Row pass.
    for r in 0..16 {
        let row_in: &[i32; 16] = input[r * 16..r * 16 + 16].try_into().unwrap();
        let row_out_mut: &mut [i32; 16] = (&mut intermediate[r * 16..r * 16 + 16])
            .try_into()
            .unwrap();
        idct16(row_in, row_out_mut);
    }

    // Column pass + residual add (ROUND_POWER_OF_TWO with shift 6).
    let mut col_in = [0i32; 16];
    let mut col_out = [0i32; 16];
    for c in 0..16 {
        for r in 0..16 {
            col_in[r] = intermediate[r * 16 + c];
        }
        idct16(&col_in, &mut col_out);
        for r in 0..16 {
            let shifted = (col_out[r] as i64 + 32) >> 6;
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], shifted);
        }
    }
}

/// 2D 1-coefficient inverse 16×16 DCT (DC-only fast path).
///
/// Mirrors `vpx_idct16x16_1_add_c` (`vpx_dsp/inv_txfm.c:799-811`). Two
/// `cospi_16_64` passes (each `wraplow`-truncated) then
/// `ROUND_POWER_OF_TWO(_, 6)`.
pub fn idct16x16_1_add(input: i32, dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 15 + 16, "dest too small for 16x16 add");

    let out1 = wraplow(dct_const_round_shift((input as i64) * (COSPI_16_64 as i64)) as i32);
    let out2 = wraplow(dct_const_round_shift((out1 as i64) * (COSPI_16_64 as i64)) as i32);
    let a1 = ((out2 as i64) + 32) >> 6;
    for r in 0..16 {
        for c in 0..16 {
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], a1);
        }
    }
}

/// 1D 32-point inverse DCT.
///
/// Mirrors libvpx `idct32_c` (`vpx_dsp/inv_txfm.c:813-1178`). Eight-stage
/// butterfly using cospi at every frequency 1..32 (excluding 0). Every
/// `step1[]`/`step2[]` store applies [`wraplow`].
#[inline]
pub fn idct32(input: &[i32; 32], output: &mut [i32; 32]) {
    let mut step1 = [0i32; 32];
    let mut step2 = [0i32; 32];

    // stage 1 — bit-reversed input load for lo 16; high-frequency cosine
    // butterflies for hi 16 (mirrors libvpx :817-873).
    step1[0] = input[0];
    step1[1] = input[16];
    step1[2] = input[8];
    step1[3] = input[24];
    step1[4] = input[4];
    step1[5] = input[20];
    step1[6] = input[12];
    step1[7] = input[28];
    step1[8] = input[2];
    step1[9] = input[18];
    step1[10] = input[10];
    step1[11] = input[26];
    step1[12] = input[6];
    step1[13] = input[22];
    step1[14] = input[14];
    step1[15] = input[30];

    let temp1 = input[1] as i64 * COSPI_31_64 as i64 - input[31] as i64 * COSPI_1_64 as i64;
    let temp2 = input[1] as i64 * COSPI_1_64 as i64 + input[31] as i64 * COSPI_31_64 as i64;
    step1[16] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[31] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = input[17] as i64 * COSPI_15_64 as i64 - input[15] as i64 * COSPI_17_64 as i64;
    let temp2 = input[17] as i64 * COSPI_17_64 as i64 + input[15] as i64 * COSPI_15_64 as i64;
    step1[17] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[30] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = input[9] as i64 * COSPI_23_64 as i64 - input[23] as i64 * COSPI_9_64 as i64;
    let temp2 = input[9] as i64 * COSPI_9_64 as i64 + input[23] as i64 * COSPI_23_64 as i64;
    step1[18] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[29] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = input[25] as i64 * COSPI_7_64 as i64 - input[7] as i64 * COSPI_25_64 as i64;
    let temp2 = input[25] as i64 * COSPI_25_64 as i64 + input[7] as i64 * COSPI_7_64 as i64;
    step1[19] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[28] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = input[5] as i64 * COSPI_27_64 as i64 - input[27] as i64 * COSPI_5_64 as i64;
    let temp2 = input[5] as i64 * COSPI_5_64 as i64 + input[27] as i64 * COSPI_27_64 as i64;
    step1[20] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[27] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = input[21] as i64 * COSPI_11_64 as i64 - input[11] as i64 * COSPI_21_64 as i64;
    let temp2 = input[21] as i64 * COSPI_21_64 as i64 + input[11] as i64 * COSPI_11_64 as i64;
    step1[21] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[26] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = input[13] as i64 * COSPI_19_64 as i64 - input[19] as i64 * COSPI_13_64 as i64;
    let temp2 = input[13] as i64 * COSPI_13_64 as i64 + input[19] as i64 * COSPI_19_64 as i64;
    step1[22] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[25] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = input[29] as i64 * COSPI_3_64 as i64 - input[3] as i64 * COSPI_29_64 as i64;
    let temp2 = input[29] as i64 * COSPI_29_64 as i64 + input[3] as i64 * COSPI_3_64 as i64;
    step1[23] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[24] = wraplow(dct_const_round_shift(temp2) as i32);

    // stage 2 (mirrors libvpx :875-920).
    step2[0] = step1[0];
    step2[1] = step1[1];
    step2[2] = step1[2];
    step2[3] = step1[3];
    step2[4] = step1[4];
    step2[5] = step1[5];
    step2[6] = step1[6];
    step2[7] = step1[7];

    let temp1 = step1[8] as i64 * COSPI_30_64 as i64 - step1[15] as i64 * COSPI_2_64 as i64;
    let temp2 = step1[8] as i64 * COSPI_2_64 as i64 + step1[15] as i64 * COSPI_30_64 as i64;
    step2[8] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[15] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step1[9] as i64 * COSPI_14_64 as i64 - step1[14] as i64 * COSPI_18_64 as i64;
    let temp2 = step1[9] as i64 * COSPI_18_64 as i64 + step1[14] as i64 * COSPI_14_64 as i64;
    step2[9] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[14] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step1[10] as i64 * COSPI_22_64 as i64 - step1[13] as i64 * COSPI_10_64 as i64;
    let temp2 = step1[10] as i64 * COSPI_10_64 as i64 + step1[13] as i64 * COSPI_22_64 as i64;
    step2[10] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[13] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step1[11] as i64 * COSPI_6_64 as i64 - step1[12] as i64 * COSPI_26_64 as i64;
    let temp2 = step1[11] as i64 * COSPI_26_64 as i64 + step1[12] as i64 * COSPI_6_64 as i64;
    step2[11] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[12] = wraplow(dct_const_round_shift(temp2) as i32);

    step2[16] = wraplow(step1[16] + step1[17]);
    step2[17] = wraplow(step1[16] - step1[17]);
    step2[18] = wraplow(-step1[18] + step1[19]);
    step2[19] = wraplow(step1[18] + step1[19]);
    step2[20] = wraplow(step1[20] + step1[21]);
    step2[21] = wraplow(step1[20] - step1[21]);
    step2[22] = wraplow(-step1[22] + step1[23]);
    step2[23] = wraplow(step1[22] + step1[23]);
    step2[24] = wraplow(step1[24] + step1[25]);
    step2[25] = wraplow(step1[24] - step1[25]);
    step2[26] = wraplow(-step1[26] + step1[27]);
    step2[27] = wraplow(step1[26] + step1[27]);
    step2[28] = wraplow(step1[28] + step1[29]);
    step2[29] = wraplow(step1[28] - step1[29]);
    step2[30] = wraplow(-step1[30] + step1[31]);
    step2[31] = wraplow(step1[30] + step1[31]);

    // stage 3 (mirrors libvpx :922-969).
    step1[0] = step2[0];
    step1[1] = step2[1];
    step1[2] = step2[2];
    step1[3] = step2[3];

    let temp1 = step2[4] as i64 * COSPI_28_64 as i64 - step2[7] as i64 * COSPI_4_64 as i64;
    let temp2 = step2[4] as i64 * COSPI_4_64 as i64 + step2[7] as i64 * COSPI_28_64 as i64;
    step1[4] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[7] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = step2[5] as i64 * COSPI_12_64 as i64 - step2[6] as i64 * COSPI_20_64 as i64;
    let temp2 = step2[5] as i64 * COSPI_20_64 as i64 + step2[6] as i64 * COSPI_12_64 as i64;
    step1[5] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[6] = wraplow(dct_const_round_shift(temp2) as i32);

    step1[8] = wraplow(step2[8] + step2[9]);
    step1[9] = wraplow(step2[8] - step2[9]);
    step1[10] = wraplow(-step2[10] + step2[11]);
    step1[11] = wraplow(step2[10] + step2[11]);
    step1[12] = wraplow(step2[12] + step2[13]);
    step1[13] = wraplow(step2[12] - step2[13]);
    step1[14] = wraplow(-step2[14] + step2[15]);
    step1[15] = wraplow(step2[14] + step2[15]);

    step1[16] = step2[16];
    step1[31] = step2[31];
    let temp1 = -(step2[17] as i64) * COSPI_4_64 as i64 + step2[30] as i64 * COSPI_28_64 as i64;
    let temp2 = step2[17] as i64 * COSPI_28_64 as i64 + step2[30] as i64 * COSPI_4_64 as i64;
    step1[17] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[30] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = -(step2[18] as i64) * COSPI_28_64 as i64 - step2[29] as i64 * COSPI_4_64 as i64;
    let temp2 = -(step2[18] as i64) * COSPI_4_64 as i64 + step2[29] as i64 * COSPI_28_64 as i64;
    step1[18] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[29] = wraplow(dct_const_round_shift(temp2) as i32);
    step1[19] = step2[19];
    step1[20] = step2[20];
    let temp1 = -(step2[21] as i64) * COSPI_20_64 as i64 + step2[26] as i64 * COSPI_12_64 as i64;
    let temp2 = step2[21] as i64 * COSPI_12_64 as i64 + step2[26] as i64 * COSPI_20_64 as i64;
    step1[21] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[26] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = -(step2[22] as i64) * COSPI_12_64 as i64 - step2[25] as i64 * COSPI_20_64 as i64;
    let temp2 = -(step2[22] as i64) * COSPI_20_64 as i64 + step2[25] as i64 * COSPI_12_64 as i64;
    step1[22] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[25] = wraplow(dct_const_round_shift(temp2) as i32);
    step1[23] = step2[23];
    step1[24] = step2[24];
    step1[27] = step2[27];
    step1[28] = step2[28];

    // stage 4 (mirrors libvpx :971-1014).
    let temp1 = (step1[0] as i64 + step1[1] as i64) * COSPI_16_64 as i64;
    let temp2 = (step1[0] as i64 - step1[1] as i64) * COSPI_16_64 as i64;
    step2[0] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[1] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = step1[2] as i64 * COSPI_24_64 as i64 - step1[3] as i64 * COSPI_8_64 as i64;
    let temp2 = step1[2] as i64 * COSPI_8_64 as i64 + step1[3] as i64 * COSPI_24_64 as i64;
    step2[2] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[3] = wraplow(dct_const_round_shift(temp2) as i32);
    step2[4] = wraplow(step1[4] + step1[5]);
    step2[5] = wraplow(step1[4] - step1[5]);
    step2[6] = wraplow(-step1[6] + step1[7]);
    step2[7] = wraplow(step1[6] + step1[7]);

    step2[8] = step1[8];
    step2[15] = step1[15];
    let temp1 = -(step1[9] as i64) * COSPI_8_64 as i64 + step1[14] as i64 * COSPI_24_64 as i64;
    let temp2 = step1[9] as i64 * COSPI_24_64 as i64 + step1[14] as i64 * COSPI_8_64 as i64;
    step2[9] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[14] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = -(step1[10] as i64) * COSPI_24_64 as i64 - step1[13] as i64 * COSPI_8_64 as i64;
    let temp2 = -(step1[10] as i64) * COSPI_8_64 as i64 + step1[13] as i64 * COSPI_24_64 as i64;
    step2[10] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[13] = wraplow(dct_const_round_shift(temp2) as i32);
    step2[11] = step1[11];
    step2[12] = step1[12];

    step2[16] = wraplow(step1[16] + step1[19]);
    step2[17] = wraplow(step1[17] + step1[18]);
    step2[18] = wraplow(step1[17] - step1[18]);
    step2[19] = wraplow(step1[16] - step1[19]);
    step2[20] = wraplow(-step1[20] + step1[23]);
    step2[21] = wraplow(-step1[21] + step1[22]);
    step2[22] = wraplow(step1[21] + step1[22]);
    step2[23] = wraplow(step1[20] + step1[23]);

    step2[24] = wraplow(step1[24] + step1[27]);
    step2[25] = wraplow(step1[25] + step1[26]);
    step2[26] = wraplow(step1[25] - step1[26]);
    step2[27] = wraplow(step1[24] - step1[27]);
    step2[28] = wraplow(-step1[28] + step1[31]);
    step2[29] = wraplow(-step1[29] + step1[30]);
    step2[30] = wraplow(step1[29] + step1[30]);
    step2[31] = wraplow(step1[28] + step1[31]);

    // stage 5 (mirrors libvpx :1016-1060).
    step1[0] = wraplow(step2[0] + step2[3]);
    step1[1] = wraplow(step2[1] + step2[2]);
    step1[2] = wraplow(step2[1] - step2[2]);
    step1[3] = wraplow(step2[0] - step2[3]);
    step1[4] = step2[4];
    let temp1 = (step2[6] as i64 - step2[5] as i64) * COSPI_16_64 as i64;
    let temp2 = (step2[5] as i64 + step2[6] as i64) * COSPI_16_64 as i64;
    step1[5] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[6] = wraplow(dct_const_round_shift(temp2) as i32);
    step1[7] = step2[7];

    step1[8] = wraplow(step2[8] + step2[11]);
    step1[9] = wraplow(step2[9] + step2[10]);
    step1[10] = wraplow(step2[9] - step2[10]);
    step1[11] = wraplow(step2[8] - step2[11]);
    step1[12] = wraplow(-step2[12] + step2[15]);
    step1[13] = wraplow(-step2[13] + step2[14]);
    step1[14] = wraplow(step2[13] + step2[14]);
    step1[15] = wraplow(step2[12] + step2[15]);

    step1[16] = step2[16];
    step1[17] = step2[17];
    let temp1 = -(step2[18] as i64) * COSPI_8_64 as i64 + step2[29] as i64 * COSPI_24_64 as i64;
    let temp2 = step2[18] as i64 * COSPI_24_64 as i64 + step2[29] as i64 * COSPI_8_64 as i64;
    step1[18] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[29] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = -(step2[19] as i64) * COSPI_8_64 as i64 + step2[28] as i64 * COSPI_24_64 as i64;
    let temp2 = step2[19] as i64 * COSPI_24_64 as i64 + step2[28] as i64 * COSPI_8_64 as i64;
    step1[19] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[28] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = -(step2[20] as i64) * COSPI_24_64 as i64 - step2[27] as i64 * COSPI_8_64 as i64;
    let temp2 = -(step2[20] as i64) * COSPI_8_64 as i64 + step2[27] as i64 * COSPI_24_64 as i64;
    step1[20] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[27] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = -(step2[21] as i64) * COSPI_24_64 as i64 - step2[26] as i64 * COSPI_8_64 as i64;
    let temp2 = -(step2[21] as i64) * COSPI_8_64 as i64 + step2[26] as i64 * COSPI_24_64 as i64;
    step1[21] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[26] = wraplow(dct_const_round_shift(temp2) as i32);
    step1[22] = step2[22];
    step1[23] = step2[23];
    step1[24] = step2[24];
    step1[25] = step2[25];
    step1[30] = step2[30];
    step1[31] = step2[31];

    // stage 6 (mirrors libvpx :1062-1100).
    step2[0] = wraplow(step1[0] + step1[7]);
    step2[1] = wraplow(step1[1] + step1[6]);
    step2[2] = wraplow(step1[2] + step1[5]);
    step2[3] = wraplow(step1[3] + step1[4]);
    step2[4] = wraplow(step1[3] - step1[4]);
    step2[5] = wraplow(step1[2] - step1[5]);
    step2[6] = wraplow(step1[1] - step1[6]);
    step2[7] = wraplow(step1[0] - step1[7]);
    step2[8] = step1[8];
    step2[9] = step1[9];
    let temp1 = (-(step1[10] as i64) + step1[13] as i64) * COSPI_16_64 as i64;
    let temp2 = (step1[10] as i64 + step1[13] as i64) * COSPI_16_64 as i64;
    step2[10] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[13] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = (-(step1[11] as i64) + step1[12] as i64) * COSPI_16_64 as i64;
    let temp2 = (step1[11] as i64 + step1[12] as i64) * COSPI_16_64 as i64;
    step2[11] = wraplow(dct_const_round_shift(temp1) as i32);
    step2[12] = wraplow(dct_const_round_shift(temp2) as i32);
    step2[14] = step1[14];
    step2[15] = step1[15];

    step2[16] = wraplow(step1[16] + step1[23]);
    step2[17] = wraplow(step1[17] + step1[22]);
    step2[18] = wraplow(step1[18] + step1[21]);
    step2[19] = wraplow(step1[19] + step1[20]);
    step2[20] = wraplow(step1[19] - step1[20]);
    step2[21] = wraplow(step1[18] - step1[21]);
    step2[22] = wraplow(step1[17] - step1[22]);
    step2[23] = wraplow(step1[16] - step1[23]);

    step2[24] = wraplow(-step1[24] + step1[31]);
    step2[25] = wraplow(-step1[25] + step1[30]);
    step2[26] = wraplow(-step1[26] + step1[29]);
    step2[27] = wraplow(-step1[27] + step1[28]);
    step2[28] = wraplow(step1[27] + step1[28]);
    step2[29] = wraplow(step1[26] + step1[29]);
    step2[30] = wraplow(step1[25] + step1[30]);
    step2[31] = wraplow(step1[24] + step1[31]);

    // stage 7 (mirrors libvpx :1102-1143).
    step1[0] = wraplow(step2[0] + step2[15]);
    step1[1] = wraplow(step2[1] + step2[14]);
    step1[2] = wraplow(step2[2] + step2[13]);
    step1[3] = wraplow(step2[3] + step2[12]);
    step1[4] = wraplow(step2[4] + step2[11]);
    step1[5] = wraplow(step2[5] + step2[10]);
    step1[6] = wraplow(step2[6] + step2[9]);
    step1[7] = wraplow(step2[7] + step2[8]);
    step1[8] = wraplow(step2[7] - step2[8]);
    step1[9] = wraplow(step2[6] - step2[9]);
    step1[10] = wraplow(step2[5] - step2[10]);
    step1[11] = wraplow(step2[4] - step2[11]);
    step1[12] = wraplow(step2[3] - step2[12]);
    step1[13] = wraplow(step2[2] - step2[13]);
    step1[14] = wraplow(step2[1] - step2[14]);
    step1[15] = wraplow(step2[0] - step2[15]);

    step1[16] = step2[16];
    step1[17] = step2[17];
    step1[18] = step2[18];
    step1[19] = step2[19];
    let temp1 = (-(step2[20] as i64) + step2[27] as i64) * COSPI_16_64 as i64;
    let temp2 = (step2[20] as i64 + step2[27] as i64) * COSPI_16_64 as i64;
    step1[20] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[27] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = (-(step2[21] as i64) + step2[26] as i64) * COSPI_16_64 as i64;
    let temp2 = (step2[21] as i64 + step2[26] as i64) * COSPI_16_64 as i64;
    step1[21] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[26] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = (-(step2[22] as i64) + step2[25] as i64) * COSPI_16_64 as i64;
    let temp2 = (step2[22] as i64 + step2[25] as i64) * COSPI_16_64 as i64;
    step1[22] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[25] = wraplow(dct_const_round_shift(temp2) as i32);
    let temp1 = (-(step2[23] as i64) + step2[24] as i64) * COSPI_16_64 as i64;
    let temp2 = (step2[23] as i64 + step2[24] as i64) * COSPI_16_64 as i64;
    step1[23] = wraplow(dct_const_round_shift(temp1) as i32);
    step1[24] = wraplow(dct_const_round_shift(temp2) as i32);
    step1[28] = step2[28];
    step1[29] = step2[29];
    step1[30] = step2[30];
    step1[31] = step2[31];

    // final stage (mirrors libvpx :1145-1177).
    output[0] = wraplow(step1[0] + step1[31]);
    output[1] = wraplow(step1[1] + step1[30]);
    output[2] = wraplow(step1[2] + step1[29]);
    output[3] = wraplow(step1[3] + step1[28]);
    output[4] = wraplow(step1[4] + step1[27]);
    output[5] = wraplow(step1[5] + step1[26]);
    output[6] = wraplow(step1[6] + step1[25]);
    output[7] = wraplow(step1[7] + step1[24]);
    output[8] = wraplow(step1[8] + step1[23]);
    output[9] = wraplow(step1[9] + step1[22]);
    output[10] = wraplow(step1[10] + step1[21]);
    output[11] = wraplow(step1[11] + step1[20]);
    output[12] = wraplow(step1[12] + step1[19]);
    output[13] = wraplow(step1[13] + step1[18]);
    output[14] = wraplow(step1[14] + step1[17]);
    output[15] = wraplow(step1[15] + step1[16]);
    output[16] = wraplow(step1[15] - step1[16]);
    output[17] = wraplow(step1[14] - step1[17]);
    output[18] = wraplow(step1[13] - step1[18]);
    output[19] = wraplow(step1[12] - step1[19]);
    output[20] = wraplow(step1[11] - step1[20]);
    output[21] = wraplow(step1[10] - step1[21]);
    output[22] = wraplow(step1[9] - step1[22]);
    output[23] = wraplow(step1[8] - step1[23]);
    output[24] = wraplow(step1[7] - step1[24]);
    output[25] = wraplow(step1[6] - step1[25]);
    output[26] = wraplow(step1[5] - step1[26]);
    output[27] = wraplow(step1[4] - step1[27]);
    output[28] = wraplow(step1[3] - step1[28]);
    output[29] = wraplow(step1[2] - step1[29]);
    output[30] = wraplow(step1[1] - step1[30]);
    output[31] = wraplow(step1[0] - step1[31]);
}

/// 2D 1024-coefficient inverse 32×32 DCT with residual add to `dest`.
///
/// `input[1024]` is the dequantised coefficients in raster order;
/// `dest[r * stride + c]` is the destination plane. Output is added to
/// `dest` and saturated to `[0, 255]`. The column pass uses
/// `ROUND_POWER_OF_TWO(_, 6)` per `vpx_idct32x32_1024_add_c`
/// (`vpx_dsp/inv_txfm.c:1180-1209`).
///
/// Implements libvpx's per-row zero-coeff fast-skip: if all 32
/// coefficients of a row are zero, the row pass is replaced by a
/// memset(0).
pub fn idct32x32_1024_add(input: &[i32; 1024], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 31 + 32, "dest too small for 32x32 add");

    let mut intermediate = [0i32; 1024];

    // Row pass with per-row zero-skip.
    for r in 0..32 {
        let row_in: &[i32; 32] = input[r * 32..r * 32 + 32].try_into().unwrap();
        // OR-reduce to detect all-zero row (matches libvpx zero_coeff check).
        let any_nonzero = row_in.iter().any(|&v| v != 0);
        let row_out_mut: &mut [i32; 32] = (&mut intermediate[r * 32..r * 32 + 32])
            .try_into()
            .unwrap();
        if any_nonzero {
            idct32(row_in, row_out_mut);
        }
        // else: leave as zero (already from initialiser).
    }

    // Column pass + residual add (ROUND_POWER_OF_TWO with shift 6).
    let mut col_in = [0i32; 32];
    let mut col_out = [0i32; 32];
    for c in 0..32 {
        for r in 0..32 {
            col_in[r] = intermediate[r * 32 + c];
        }
        idct32(&col_in, &mut col_out);
        for r in 0..32 {
            let shifted = (col_out[r] as i64 + 32) >> 6;
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], shifted);
        }
    }
}

/// 2D 1-coefficient inverse 32×32 DCT (DC-only fast path).
///
/// Mirrors `vpx_idct32x32_1_add_c` (`vpx_dsp/inv_txfm.c:1263-1276`). Two
/// `cospi_16_64` passes (each `wraplow`-truncated) then
/// `ROUND_POWER_OF_TWO(_, 6)`.
pub fn idct32x32_1_add(input: i32, dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 31 + 32, "dest too small for 32x32 add");

    let out1 = wraplow(dct_const_round_shift((input as i64) * (COSPI_16_64 as i64)) as i32);
    let out2 = wraplow(dct_const_round_shift((out1 as i64) * (COSPI_16_64 as i64)) as i32);
    let a1 = ((out2 as i64) + 32) >> 6;
    for r in 0..32 {
        for c in 0..32 {
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], a1);
        }
    }
}

/// 1D 4-point inverse ADST (asymmetric discrete sine transform).
///
/// Mirrors libvpx `iadst4_c` (`vpx_dsp/inv_txfm.c:96-131`). Uses the
/// `sinpi_*_9` constants. The all-zero early-out short-circuit from
/// libvpx is preserved.
#[inline]
pub fn iadst4(input: &[i32; 4], output: &mut [i32; 4]) {
    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];

    if (x0 | x1 | x2 | x3) == 0 {
        output.fill(0);
        return;
    }

    // Stage 1 multiplications — products fit comfortably in i64.
    let s0_a = SINPI_1_9 as i64 * x0 as i64;
    let s1_a = SINPI_2_9 as i64 * x0 as i64;
    let s2_a = SINPI_3_9 as i64 * x1 as i64;
    let s3_a = SINPI_4_9 as i64 * x2 as i64;
    let s4_a = SINPI_1_9 as i64 * x2 as i64;
    let s5_a = SINPI_2_9 as i64 * x3 as i64;
    let s6_a = SINPI_4_9 as i64 * x3 as i64;
    // libvpx: s7 = WRAPLOW(x0 - x2 + x3); kept as i32 (then sinpi-mul to i64).
    let s7 = wraplow(x0 - x2 + x3);

    let s0 = s0_a + s3_a + s5_a;
    let s1 = s1_a - s4_a - s6_a;
    let s3 = s2_a;
    let s2 = SINPI_3_9 as i64 * s7 as i64;

    // 1-D scaling factor sqrt(2); output truncated to i16 via wraplow.
    output[0] = wraplow(dct_const_round_shift(s0 + s3) as i32);
    output[1] = wraplow(dct_const_round_shift(s1 + s3) as i32);
    output[2] = wraplow(dct_const_round_shift(s2) as i32);
    output[3] = wraplow(dct_const_round_shift(s0 + s1 - s3) as i32);
}

/// 1D 8-point inverse ADST.
///
/// Mirrors libvpx `iadst8_c` (`vpx_dsp/inv_txfm.c:196-269`). Three-stage
/// butterfly using cospi_*_64 at the 2/6/10/14/18/22/26/30 (stage 1),
/// 8/24 (stage 2), and 16 (stage 3) frequencies, plus a final
/// permutation+sign-flip pattern on the output. The all-zero early-out
/// is preserved.
#[inline]
pub fn iadst8(input: &[i32; 8], output: &mut [i32; 8]) {
    let mut x0: i32 = input[7];
    let mut x1: i32 = input[0];
    let mut x2: i32 = input[5];
    let mut x3: i32 = input[2];
    let mut x4: i32 = input[3];
    let mut x5: i32 = input[4];
    let mut x6: i32 = input[1];
    let mut x7: i32 = input[6];

    if (x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7) == 0 {
        output.fill(0);
        return;
    }

    // Stage 1 — libvpx assigns the i64 products to `int` s-vars, which
    // implicitly truncates to i32. Mirror with `as i32`.
    let s0 = (COSPI_2_64 as i64 * x0 as i64 + COSPI_30_64 as i64 * x1 as i64) as i32;
    let s1 = (COSPI_30_64 as i64 * x0 as i64 - COSPI_2_64 as i64 * x1 as i64) as i32;
    let s2 = (COSPI_10_64 as i64 * x2 as i64 + COSPI_22_64 as i64 * x3 as i64) as i32;
    let s3 = (COSPI_22_64 as i64 * x2 as i64 - COSPI_10_64 as i64 * x3 as i64) as i32;
    let s4 = (COSPI_18_64 as i64 * x4 as i64 + COSPI_14_64 as i64 * x5 as i64) as i32;
    let s5 = (COSPI_14_64 as i64 * x4 as i64 - COSPI_18_64 as i64 * x5 as i64) as i32;
    let s6 = (COSPI_26_64 as i64 * x6 as i64 + COSPI_6_64 as i64 * x7 as i64) as i32;
    let s7 = (COSPI_6_64 as i64 * x6 as i64 - COSPI_26_64 as i64 * x7 as i64) as i32;

    x0 = wraplow(dct_const_round_shift(s0 as i64 + s4 as i64) as i32);
    x1 = wraplow(dct_const_round_shift(s1 as i64 + s5 as i64) as i32);
    x2 = wraplow(dct_const_round_shift(s2 as i64 + s6 as i64) as i32);
    x3 = wraplow(dct_const_round_shift(s3 as i64 + s7 as i64) as i32);
    x4 = wraplow(dct_const_round_shift(s0 as i64 - s4 as i64) as i32);
    x5 = wraplow(dct_const_round_shift(s1 as i64 - s5 as i64) as i32);
    x6 = wraplow(dct_const_round_shift(s2 as i64 - s6 as i64) as i32);
    x7 = wraplow(dct_const_round_shift(s3 as i64 - s7 as i64) as i32);

    // Stage 2 — same i64→i32 truncation pattern.
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = (COSPI_8_64 as i64 * x4 as i64 + COSPI_24_64 as i64 * x5 as i64) as i32;
    let s5 = (COSPI_24_64 as i64 * x4 as i64 - COSPI_8_64 as i64 * x5 as i64) as i32;
    let s6 = (-(COSPI_24_64 as i64) * x6 as i64 + COSPI_8_64 as i64 * x7 as i64) as i32;
    let s7 = (COSPI_8_64 as i64 * x6 as i64 + COSPI_24_64 as i64 * x7 as i64) as i32;

    x0 = wraplow(s0 + s2);
    x1 = wraplow(s1 + s3);
    x2 = wraplow(s0 - s2);
    x3 = wraplow(s1 - s3);
    x4 = wraplow(dct_const_round_shift(s4 as i64 + s6 as i64) as i32);
    x5 = wraplow(dct_const_round_shift(s5 as i64 + s7 as i64) as i32);
    x6 = wraplow(dct_const_round_shift(s4 as i64 - s6 as i64) as i32);
    x7 = wraplow(dct_const_round_shift(s5 as i64 - s7 as i64) as i32);

    // Stage 3.
    let s2 = (COSPI_16_64 as i64 * (x2 as i64 + x3 as i64)) as i32;
    let s3 = (COSPI_16_64 as i64 * (x2 as i64 - x3 as i64)) as i32;
    let s6 = (COSPI_16_64 as i64 * (x6 as i64 + x7 as i64)) as i32;
    let s7 = (COSPI_16_64 as i64 * (x6 as i64 - x7 as i64)) as i32;

    x2 = wraplow(dct_const_round_shift(s2 as i64) as i32);
    x3 = wraplow(dct_const_round_shift(s3 as i64) as i32);
    x6 = wraplow(dct_const_round_shift(s6 as i64) as i32);
    x7 = wraplow(dct_const_round_shift(s7 as i64) as i32);

    // Output permutation + sign flips (mirrors libvpx :261-268).
    output[0] = wraplow(x0);
    output[1] = wraplow(-x4);
    output[2] = wraplow(x6);
    output[3] = wraplow(-x2);
    output[4] = wraplow(x3);
    output[5] = wraplow(-x7);
    output[6] = wraplow(x5);
    output[7] = wraplow(-x1);
}

/// 1D 16-point inverse ADST.
///
/// Mirrors libvpx `iadst16_c` (`vpx_dsp/inv_txfm.c:389-555`). Four-stage
/// butterfly using cospi at frequencies 1/3/5/7/9/11/13/15/17/19/21/23/
/// 25/27/29/31 (stage 1), 4/12/20/28 (stage 2), 8/24 (stage 3),
/// 16 (stage 4), plus output permutation with sign flips.
#[inline]
pub fn iadst16(input: &[i32; 16], output: &mut [i32; 16]) {
    let mut x0: i32 = input[15];
    let mut x1: i32 = input[0];
    let mut x2: i32 = input[13];
    let mut x3: i32 = input[2];
    let mut x4: i32 = input[11];
    let mut x5: i32 = input[4];
    let mut x6: i32 = input[9];
    let mut x7: i32 = input[6];
    let mut x8: i32 = input[7];
    let mut x9: i32 = input[8];
    let mut x10: i32 = input[5];
    let mut x11: i32 = input[10];
    let mut x12: i32 = input[3];
    let mut x13: i32 = input[12];
    let mut x14: i32 = input[1];
    let mut x15: i32 = input[14];

    if (x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15)
        == 0
    {
        output.fill(0);
        return;
    }

    // Stage 1.
    let s0 = x0 as i64 * COSPI_1_64 as i64 + x1 as i64 * COSPI_31_64 as i64;
    let s1 = x0 as i64 * COSPI_31_64 as i64 - x1 as i64 * COSPI_1_64 as i64;
    let s2 = x2 as i64 * COSPI_5_64 as i64 + x3 as i64 * COSPI_27_64 as i64;
    let s3 = x2 as i64 * COSPI_27_64 as i64 - x3 as i64 * COSPI_5_64 as i64;
    let s4 = x4 as i64 * COSPI_9_64 as i64 + x5 as i64 * COSPI_23_64 as i64;
    let s5 = x4 as i64 * COSPI_23_64 as i64 - x5 as i64 * COSPI_9_64 as i64;
    let s6 = x6 as i64 * COSPI_13_64 as i64 + x7 as i64 * COSPI_19_64 as i64;
    let s7 = x6 as i64 * COSPI_19_64 as i64 - x7 as i64 * COSPI_13_64 as i64;
    let s8 = x8 as i64 * COSPI_17_64 as i64 + x9 as i64 * COSPI_15_64 as i64;
    let s9 = x8 as i64 * COSPI_15_64 as i64 - x9 as i64 * COSPI_17_64 as i64;
    let s10 = x10 as i64 * COSPI_21_64 as i64 + x11 as i64 * COSPI_11_64 as i64;
    let s11 = x10 as i64 * COSPI_11_64 as i64 - x11 as i64 * COSPI_21_64 as i64;
    let s12 = x12 as i64 * COSPI_25_64 as i64 + x13 as i64 * COSPI_7_64 as i64;
    let s13 = x12 as i64 * COSPI_7_64 as i64 - x13 as i64 * COSPI_25_64 as i64;
    let s14 = x14 as i64 * COSPI_29_64 as i64 + x15 as i64 * COSPI_3_64 as i64;
    let s15 = x14 as i64 * COSPI_3_64 as i64 - x15 as i64 * COSPI_29_64 as i64;

    x0 = wraplow(dct_const_round_shift(s0 + s8) as i32);
    x1 = wraplow(dct_const_round_shift(s1 + s9) as i32);
    x2 = wraplow(dct_const_round_shift(s2 + s10) as i32);
    x3 = wraplow(dct_const_round_shift(s3 + s11) as i32);
    x4 = wraplow(dct_const_round_shift(s4 + s12) as i32);
    x5 = wraplow(dct_const_round_shift(s5 + s13) as i32);
    x6 = wraplow(dct_const_round_shift(s6 + s14) as i32);
    x7 = wraplow(dct_const_round_shift(s7 + s15) as i32);
    x8 = wraplow(dct_const_round_shift(s0 - s8) as i32);
    x9 = wraplow(dct_const_round_shift(s1 - s9) as i32);
    x10 = wraplow(dct_const_round_shift(s2 - s10) as i32);
    x11 = wraplow(dct_const_round_shift(s3 - s11) as i32);
    x12 = wraplow(dct_const_round_shift(s4 - s12) as i32);
    x13 = wraplow(dct_const_round_shift(s5 - s13) as i32);
    x14 = wraplow(dct_const_round_shift(s6 - s14) as i32);
    x15 = wraplow(dct_const_round_shift(s7 - s15) as i32);

    // Stage 2.
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = x4;
    let s5 = x5;
    let s6 = x6;
    let s7 = x7;
    let s8 = x8 as i64 * COSPI_4_64 as i64 + x9 as i64 * COSPI_28_64 as i64;
    let s9 = x8 as i64 * COSPI_28_64 as i64 - x9 as i64 * COSPI_4_64 as i64;
    let s10 = x10 as i64 * COSPI_20_64 as i64 + x11 as i64 * COSPI_12_64 as i64;
    let s11 = x10 as i64 * COSPI_12_64 as i64 - x11 as i64 * COSPI_20_64 as i64;
    let s12 = -(x12 as i64) * COSPI_28_64 as i64 + x13 as i64 * COSPI_4_64 as i64;
    let s13 = x12 as i64 * COSPI_4_64 as i64 + x13 as i64 * COSPI_28_64 as i64;
    let s14 = -(x14 as i64) * COSPI_12_64 as i64 + x15 as i64 * COSPI_20_64 as i64;
    let s15 = x14 as i64 * COSPI_20_64 as i64 + x15 as i64 * COSPI_12_64 as i64;

    x0 = wraplow(s0 + s4);
    x1 = wraplow(s1 + s5);
    x2 = wraplow(s2 + s6);
    x3 = wraplow(s3 + s7);
    x4 = wraplow(s0 - s4);
    x5 = wraplow(s1 - s5);
    x6 = wraplow(s2 - s6);
    x7 = wraplow(s3 - s7);
    x8 = wraplow(dct_const_round_shift(s8 + s12) as i32);
    x9 = wraplow(dct_const_round_shift(s9 + s13) as i32);
    x10 = wraplow(dct_const_round_shift(s10 + s14) as i32);
    x11 = wraplow(dct_const_round_shift(s11 + s15) as i32);
    x12 = wraplow(dct_const_round_shift(s8 - s12) as i32);
    x13 = wraplow(dct_const_round_shift(s9 - s13) as i32);
    x14 = wraplow(dct_const_round_shift(s10 - s14) as i32);
    x15 = wraplow(dct_const_round_shift(s11 - s15) as i32);

    // Stage 3.
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = x4 as i64 * COSPI_8_64 as i64 + x5 as i64 * COSPI_24_64 as i64;
    let s5 = x4 as i64 * COSPI_24_64 as i64 - x5 as i64 * COSPI_8_64 as i64;
    let s6 = -(x6 as i64) * COSPI_24_64 as i64 + x7 as i64 * COSPI_8_64 as i64;
    let s7 = x6 as i64 * COSPI_8_64 as i64 + x7 as i64 * COSPI_24_64 as i64;
    let s8 = x8;
    let s9 = x9;
    let s10 = x10;
    let s11 = x11;
    let s12 = x12 as i64 * COSPI_8_64 as i64 + x13 as i64 * COSPI_24_64 as i64;
    let s13 = x12 as i64 * COSPI_24_64 as i64 - x13 as i64 * COSPI_8_64 as i64;
    let s14 = -(x14 as i64) * COSPI_24_64 as i64 + x15 as i64 * COSPI_8_64 as i64;
    let s15 = x14 as i64 * COSPI_8_64 as i64 + x15 as i64 * COSPI_24_64 as i64;

    x0 = wraplow(s0 + s2);
    x1 = wraplow(s1 + s3);
    x2 = wraplow(s0 - s2);
    x3 = wraplow(s1 - s3);
    x4 = wraplow(dct_const_round_shift(s4 + s6) as i32);
    x5 = wraplow(dct_const_round_shift(s5 + s7) as i32);
    x6 = wraplow(dct_const_round_shift(s4 - s6) as i32);
    x7 = wraplow(dct_const_round_shift(s5 - s7) as i32);
    x8 = wraplow(s8 + s10);
    x9 = wraplow(s9 + s11);
    x10 = wraplow(s8 - s10);
    x11 = wraplow(s9 - s11);
    x12 = wraplow(dct_const_round_shift(s12 + s14) as i32);
    x13 = wraplow(dct_const_round_shift(s13 + s15) as i32);
    x14 = wraplow(dct_const_round_shift(s12 - s14) as i32);
    x15 = wraplow(dct_const_round_shift(s13 - s15) as i32);

    // Stage 4.
    let s2 = -(COSPI_16_64 as i64) * (x2 as i64 + x3 as i64);
    let s3 = COSPI_16_64 as i64 * (x2 as i64 - x3 as i64);
    let s6 = COSPI_16_64 as i64 * (x6 as i64 + x7 as i64);
    let s7 = COSPI_16_64 as i64 * (-(x6 as i64) + x7 as i64);
    let s10 = COSPI_16_64 as i64 * (x10 as i64 + x11 as i64);
    let s11 = COSPI_16_64 as i64 * (-(x10 as i64) + x11 as i64);
    let s14 = -(COSPI_16_64 as i64) * (x14 as i64 + x15 as i64);
    let s15 = COSPI_16_64 as i64 * (x14 as i64 - x15 as i64);

    x2 = wraplow(dct_const_round_shift(s2) as i32);
    x3 = wraplow(dct_const_round_shift(s3) as i32);
    x6 = wraplow(dct_const_round_shift(s6) as i32);
    x7 = wraplow(dct_const_round_shift(s7) as i32);
    x10 = wraplow(dct_const_round_shift(s10) as i32);
    x11 = wraplow(dct_const_round_shift(s11) as i32);
    x14 = wraplow(dct_const_round_shift(s14) as i32);
    x15 = wraplow(dct_const_round_shift(s15) as i32);

    // Output permutation + sign flips (mirrors libvpx :539-554).
    output[0] = wraplow(x0);
    output[1] = wraplow(-x8);
    output[2] = wraplow(x12);
    output[3] = wraplow(-x4);
    output[4] = wraplow(x6);
    output[5] = wraplow(x14);
    output[6] = wraplow(x10);
    output[7] = wraplow(x2);
    output[8] = wraplow(x3);
    output[9] = wraplow(x11);
    output[10] = wraplow(x15);
    output[11] = wraplow(x7);
    output[12] = wraplow(x5);
    output[13] = wraplow(-x13);
    output[14] = wraplow(x9);
    output[15] = wraplow(-x1);
}

/// libvpx `UNIT_QUANT_SHIFT` (`vpx_dsp/txfm_common.h:20`). The
/// 4×4 inverse Walsh-Hadamard right-shifts every input coefficient by
/// this amount up-front; the matching forward kernel scales coefficients
/// by `UNIT_QUANT_FACTOR = 1 << UNIT_QUANT_SHIFT = 4` so the round-trip
/// cancels.
///
/// G2 (decoder-hardening V6 §G2): scoped to this module because only the
/// WHT kernels reference it; if a forward WHT lands later it should
/// re-export from here.
const UNIT_QUANT_SHIFT: u32 = 2;

/// 2D 16-coefficient inverse 4×4 Walsh-Hadamard transform with residual
/// add to `dest` (lossless mode).
///
/// `input` is the 16 dequantised coefficients in raster order. `dest`
/// is the prediction (or the zero-base initial buffer); the transform
/// output is added pixel-by-pixel and saturated to `[0, 255]`.
///
/// VP9 lossless mode (`lossless == 1` in the frame header) uses this
/// kernel instead of the standard 4×4 inverse DCT at every `TX_4X4`
/// position. Larger TX sizes are not legal in lossless mode (libvpx
/// `vp9_decodeframe.c:280` forces `tx_size = TX_4X4` when lossless).
///
/// Mirrors `vpx_iwht4x4_16_add_c` (`vpx_dsp/inv_txfm.c:18-68`)
/// byte-for-byte: a 4-point reversible orthonormal Walsh-Hadamard in
/// 3.5 adds and 0.5 shifts per pixel, applied first as a row pass then
/// a column pass + residual add.
pub fn iwht4x4_16_add(input: &[i32; 16], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 3 + 4, "dest too small for 4x4 add");

    let mut output = [0i32; 16];

    // Row pass — read `[a, c, d, b] = ip[0..4]`, divide by
    // UNIT_QUANT_FACTOR=4 (the encoder-side WHT inflates by this), run
    // the butterfly, then WRAPLOW-store back as `[a, b, c, d]`.
    for r in 0..4 {
        let ip = &input[r * 4..r * 4 + 4];
        let mut a1: i64 = (ip[0] as i64) >> UNIT_QUANT_SHIFT;
        let mut c1: i64 = (ip[1] as i64) >> UNIT_QUANT_SHIFT;
        let mut d1: i64 = (ip[2] as i64) >> UNIT_QUANT_SHIFT;
        let mut b1: i64 = (ip[3] as i64) >> UNIT_QUANT_SHIFT;
        a1 += c1;
        d1 -= b1;
        let e1: i64 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        let op = &mut output[r * 4..r * 4 + 4];
        op[0] = wraplow(a1 as i32);
        op[1] = wraplow(b1 as i32);
        op[2] = wraplow(c1 as i32);
        op[3] = wraplow(d1 as i32);
    }

    // Column pass + residual add. libvpx walks the columns by stepping
    // `ip` one slot per column and reading `ip[4*r]` for each row; we
    // index directly. No further shift on the output — `clip_pixel_add`
    // does the WRAPLOW + clamp.
    for c in 0..4 {
        let mut a1: i64 = output[c] as i64;
        let mut c1: i64 = output[4 + c] as i64;
        let mut d1: i64 = output[8 + c] as i64;
        let mut b1: i64 = output[12 + c] as i64;
        a1 += c1;
        d1 -= b1;
        let e1: i64 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        dest[c] = clip_pixel_add(dest[c], wraplow(a1 as i32) as i64);
        dest[stride + c] = clip_pixel_add(dest[stride + c], wraplow(b1 as i32) as i64);
        dest[2 * stride + c] = clip_pixel_add(dest[2 * stride + c], wraplow(c1 as i32) as i64);
        dest[3 * stride + c] = clip_pixel_add(dest[3 * stride + c], wraplow(d1 as i32) as i64);
    }
}

/// 2D 1-coefficient inverse 4×4 Walsh-Hadamard fast path (DC-only).
///
/// `input` is the DC coefficient (`coef[0]`); the remaining 15 are
/// assumed zero. Produces a result bit-identical to
/// [`iwht4x4_16_add`] for any DC-only input — see the
/// `iwht4x4_1_add_matches_16_add_for_dc_only` unit test.
///
/// Mirrors `vpx_iwht4x4_1_add_c` (`vpx_dsp/inv_txfm.c:70-94`).
pub fn iwht4x4_1_add(input: i32, dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 3 + 4, "dest too small for 4x4 add");

    // Row pass: only `ip[0]` is non-zero, so `c1 = d1 = b1 = 0` and the
    // recurrence collapses to `e1 = a1 >> 1; a1 -= e1`, writing
    // `[a1, e1, e1, e1]`. libvpx allocates a `tran_low_t tmp[4]`; we
    // inline the four values.
    let a1: i64 = (input as i64) >> UNIT_QUANT_SHIFT;
    let e1: i64 = a1 >> 1;
    let a1: i64 = a1 - e1;
    let tmp_a = wraplow(a1 as i32);
    let tmp_e = wraplow(e1 as i32);
    let tmp = [tmp_a, tmp_e, tmp_e, tmp_e];

    // Column pass: each column starts with `ip[0] = tmp[c]`,
    // `ip[4] = ip[8] = ip[12] = 0`. Recurrence again collapses to
    // `e1 = ip[0] >> 1; a1 = ip[0] - e1`, writing
    // `[a1, e1, e1, e1]` down the column. libvpx does NOT WRAPLOW the
    // per-pixel add inside the loop, and neither do we: values produced
    // here are small enough (DC pre-scaled by `>> UNIT_QUANT_SHIFT` = 4)
    // that WRAPLOW (`x as i16 as i32`) would be a no-op in any realistic
    // range, so libvpx — and we — match by omitting it.
    for c in 0..4 {
        let ip0: i64 = tmp[c] as i64;
        let e1: i64 = ip0 >> 1;
        let a1: i64 = ip0 - e1;
        dest[c] = clip_pixel_add(dest[c], a1);
        dest[stride + c] = clip_pixel_add(dest[stride + c], e1);
        dest[2 * stride + c] = clip_pixel_add(dest[2 * stride + c], e1);
        dest[3 * stride + c] = clip_pixel_add(dest[3 * stride + c], e1);
    }
}

/// Function-pointer type for a fixed-size 1D inverse transform.
type Transform1D<const N: usize> = fn(&[i32; N], &mut [i32; N]);

/// `(row_xform, col_xform)` pair returned by an `iht_N_dispatch`.
type TransformPair<const N: usize> = (Transform1D<N>, Transform1D<N>);

/// Return the (row_xform, col_xform) function pair for an N=4 hybrid
/// transform. The libvpx `transform_2d` struct lays out `rows` first,
/// then `cols`; we follow.
#[inline]
fn iht_4_dispatch(tx_type: TxType) -> TransformPair<4> {
    // Returns `(row_xform, col_xform)`.
    //
    // libvpx's `transform_2d` struct is `{ transform_1d cols, rows; }` —
    // `cols` is the FIRST field. So `IHT_4[ADST_DCT] = { iadst4_c, idct4_c }`
    // means `cols=iadst4`, `rows=idct4`, which is `(row=idct4, col=iadst4)`
    // in our tuple order. Stage 3 stretch M.8 caught this — the original
    // dispatch mis-interpreted the brace-initializer order as
    // `{rows, cols}` (the natural reading order in English), and the M.3
    // reference port made the same mistake, so the N=100 differential
    // couldn't catch it. See M.8 oracle-diff binary for the localization.
    match tx_type {
        TxType::DctDct => (idct4, idct4),
        TxType::AdstDct => (idct4, iadst4),
        TxType::DctAdst => (iadst4, idct4),
        TxType::AdstAdst => (iadst4, iadst4),
    }
}

#[inline]
fn iht_8_dispatch(tx_type: TxType) -> TransformPair<8> {
    // See note on [`iht_4_dispatch`] — libvpx struct order is `{cols, rows}`.
    match tx_type {
        TxType::DctDct => (idct8, idct8),
        TxType::AdstDct => (idct8, iadst8),
        TxType::DctAdst => (iadst8, idct8),
        TxType::AdstAdst => (iadst8, iadst8),
    }
}

#[inline]
fn iht_16_dispatch(tx_type: TxType) -> TransformPair<16> {
    // See note on [`iht_4_dispatch`] — libvpx struct order is `{cols, rows}`.
    match tx_type {
        TxType::DctDct => (idct16, idct16),
        TxType::AdstDct => (idct16, iadst16),
        TxType::DctAdst => (iadst16, idct16),
        TxType::AdstAdst => (iadst16, iadst16),
    }
}

/// 2D 4×4 hybrid inverse transform with residual add.
///
/// Per `tx_type` (libvpx struct `transform_2d { cols, rows }` — `cols`
/// is the first field, see Stage 3 stretch M.8 note on
/// [`iht_4_dispatch`]):
/// - `DctDct`: row=idct4, col=idct4 (equivalent to [`idct4x4_16_add`]).
/// - `AdstDct`: row=idct4, col=iadst4.
/// - `DctAdst`: row=iadst4, col=idct4.
/// - `AdstAdst`: row=iadst4, col=iadst4.
///
/// Column-pass shift is `ROUND_POWER_OF_TWO(_, 4)` per libvpx
/// `vp9/common/vp9_idct.c:46`.
///
/// Mirrors `vp9_iht4x4_16_add_c` (`vp9/common/vp9_idct.c:20-50`).
pub fn iht4x4_16_add(input: &[i32; 16], dest: &mut [u8], stride: usize, tx_type: TxType) {
    debug_assert!(dest.len() >= stride * 3 + 4, "dest too small for 4x4 add");

    let (row_xform, col_xform) = iht_4_dispatch(tx_type);
    let mut intermediate = [0i32; 16];

    // Row pass.
    for r in 0..4 {
        let row_in: &[i32; 4] = input[r * 4..r * 4 + 4].try_into().unwrap();
        let row_out_mut: &mut [i32; 4] = (&mut intermediate[r * 4..r * 4 + 4])
            .try_into()
            .unwrap();
        row_xform(row_in, row_out_mut);
    }

    // Column pass + residual add.
    for c in 0..4 {
        let col_in: [i32; 4] = [
            intermediate[c],
            intermediate[4 + c],
            intermediate[8 + c],
            intermediate[12 + c],
        ];
        let mut col_out = [0i32; 4];
        col_xform(&col_in, &mut col_out);
        for r in 0..4 {
            let shifted = (col_out[r] as i64 + 8) >> 4;
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], shifted);
        }
    }
}

/// 2D 8×8 hybrid inverse transform with residual add.
///
/// Same dispatch pattern as [`iht4x4_16_add`] at the 8×8 size; column
/// shift is `ROUND_POWER_OF_TWO(_, 5)`.
///
/// Mirrors `vp9_iht8x8_64_add_c` (`vp9/common/vp9_idct.c:59-83`).
pub fn iht8x8_64_add(input: &[i32; 64], dest: &mut [u8], stride: usize, tx_type: TxType) {
    debug_assert!(dest.len() >= stride * 7 + 8, "dest too small for 8x8 add");

    let (row_xform, col_xform) = iht_8_dispatch(tx_type);
    let mut intermediate = [0i32; 64];

    for r in 0..8 {
        let row_in: &[i32; 8] = input[r * 8..r * 8 + 8].try_into().unwrap();
        let row_out_mut: &mut [i32; 8] = (&mut intermediate[r * 8..r * 8 + 8])
            .try_into()
            .unwrap();
        row_xform(row_in, row_out_mut);
    }

    let mut col_in = [0i32; 8];
    let mut col_out = [0i32; 8];
    for c in 0..8 {
        for r in 0..8 {
            col_in[r] = intermediate[r * 8 + c];
        }
        col_xform(&col_in, &mut col_out);
        for r in 0..8 {
            let shifted = (col_out[r] as i64 + 16) >> 5;
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], shifted);
        }
    }
}

/// 2D 16×16 hybrid inverse transform with residual add.
///
/// Same dispatch pattern as [`iht4x4_16_add`] at the 16×16 size; column
/// shift is `ROUND_POWER_OF_TWO(_, 6)`.
///
/// Mirrors `vp9_iht16x16_256_add_c` (`vp9/common/vp9_idct.c:92-116`).
pub fn iht16x16_256_add(input: &[i32; 256], dest: &mut [u8], stride: usize, tx_type: TxType) {
    debug_assert!(dest.len() >= stride * 15 + 16, "dest too small for 16x16 add");

    let (row_xform, col_xform) = iht_16_dispatch(tx_type);
    let mut intermediate = [0i32; 256];

    for r in 0..16 {
        let row_in: &[i32; 16] = input[r * 16..r * 16 + 16].try_into().unwrap();
        let row_out_mut: &mut [i32; 16] = (&mut intermediate[r * 16..r * 16 + 16])
            .try_into()
            .unwrap();
        row_xform(row_in, row_out_mut);
    }

    let mut col_in = [0i32; 16];
    let mut col_out = [0i32; 16];
    for c in 0..16 {
        for r in 0..16 {
            col_in[r] = intermediate[r * 16 + c];
        }
        col_xform(&col_in, &mut col_out);
        for r in 0..16 {
            let shifted = (col_out[r] as i64 + 32) >> 6;
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], shifted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero input → zero output (1D).
    #[test]
    fn idct4_zero_in_zero_out() {
        let input = [0i32; 4];
        let mut output = [99i32; 4];
        idct4(&input, &mut output);
        assert_eq!(output, [0; 4]);
    }

    /// Stage 3 stretch M.13 probe: capture the V-block (24,12) iht4x4
    /// output for the 32×32 vector frame 1 cascade analysis. With above
    /// = [88, 96, 101, 109] (V mode → every row replicates above) and
    /// the captured dq_coefs (eob=11), prints the post-add destination.
    /// Not an assertion — manually compared against vpxdec post-LF +
    /// the libvpx oracle iht run.
    #[test]
    fn m13_v_block_24_12_iht4x4_probe() {
        let dq_coefs: [i32; 16] =
            [0, 231, -231, 231, -231, 0, 231, 0, 0, 0, 0, 0, 231, 0, 0, 0];
        let mut dst = [
            88u8, 96, 101, 109,
            88,   96, 101, 109,
            88,   96, 101, 109,
            88,   96, 101, 109,
        ];
        iht4x4_16_add(&dq_coefs, &mut dst, 4, TxType::AdstDct);
        eprintln!("M13 V-block (24,12) iht4x4 output:");
        for r in 0..4 {
            eprint!("  row {}: ", r);
            for c in 0..4 {
                eprint!("{:3} ", dst[r * 4 + c]);
            }
            eprintln!();
        }
        eprintln!(
            "M13 row 3 col 3 (= pre-LF col=27, row=15) = {}; \
             trace says it should be 79",
            dst[15]
        );
    }

    /// Zero input → unchanged dest (2D, since add of 0+rounding).
    /// `(0 + 8) >> 4 = 0`, so dest is preserved.
    #[test]
    fn idct4x4_16_add_zero_preserves_dest() {
        let input = [0i32; 16];
        let mut dest = [128u8; 16]; // 4x4 block, stride 4
        idct4x4_16_add(&input, &mut dest, 4);
        assert_eq!(dest, [128; 16]);
    }

    /// DC-only path matches the full path when only `input[0]` is set.
    #[test]
    fn idct4x4_1_add_matches_16_add_for_dc_only() {
        let dc: i32 = 200;
        let mut input = [0i32; 16];
        input[0] = dc;
        let mut dest_full = [50u8; 16];
        let mut dest_dc = [50u8; 16];
        idct4x4_16_add(&input, &mut dest_full, 4);
        idct4x4_1_add(dc, &mut dest_dc, 4);
        assert_eq!(
            dest_full, dest_dc,
            "DC-only fast path must match full path for DC-only input"
        );
    }

    /// Differential against a hand-computed reference for a known
    /// non-trivial coefficient input. The expected values come from
    /// running libvpx's `vpx_idct4x4_16_add_c` on the same input with
    /// `dest = [0; 16]`, captured by hand.
    ///
    /// Input is `[1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`:
    /// a single DC coefficient of 1024.
    ///
    /// Calculation:
    /// - idct4 of [1024, 0, 0, 0]: step0 = step1 = round_shift(1024 * 11585)
    ///   = round_shift(11862784) = (11862784 + 8192) >> 14 = 11870976 >> 14
    ///   = 724. step2=step3=0. Output = [724, 724, 724, 724].
    /// - After row pass, intermediate is 4 rows of [724, 724, 724, 724]
    ///   stored at offsets 0, 4, 8, 12 — but only row 0 has nonzero data
    ///   since rows 1..3 had all-zero inputs. So intermediate =
    ///   [724,724,724,724, 0,0,0,0, 0,0,0,0, 0,0,0,0].
    /// - Column pass on column c reads intermediate[c, c+4, c+8, c+12]
    ///   = [724, 0, 0, 0]. idct4([724,0,0,0]) = step0=step1=round_shift(724*11585)
    ///   = round_shift(8387540) = (8387540+8192)>>14 = 8395732>>14 = 512.
    ///   Output = [512, 512, 512, 512].
    /// - Final add: (512 + 8) >> 4 = 520 >> 4 = 32. Add 32 to each
    ///   pixel of dest=0, giving 32.
    #[test]
    fn idct4x4_16_add_dc_known_value() {
        let mut input = [0i32; 16];
        input[0] = 1024;
        let mut dest = [0u8; 16];
        idct4x4_16_add(&input, &mut dest, 4);
        assert_eq!(dest, [32u8; 16]);
    }

    /// Saturation: large positive DC should clip pixels at 255.
    #[test]
    fn idct4x4_16_add_saturates_high() {
        let mut input = [0i32; 16];
        input[0] = 30_000;
        let mut dest = [200u8; 16];
        idct4x4_16_add(&input, &mut dest, 4);
        // 30000 → after two cospi_16_64 passes + >>4, well over 255.
        assert_eq!(dest, [255; 16]);
    }

    /// Saturation: large negative DC should clip pixels at 0.
    #[test]
    fn idct4x4_16_add_saturates_low() {
        let mut input = [0i32; 16];
        input[0] = -30_000;
        let mut dest = [100u8; 16];
        idct4x4_16_add(&input, &mut dest, 4);
        assert_eq!(dest, [0; 16]);
    }

    /// Non-4-stride dest works (the function adds to `dest[r * stride + c]`).
    #[test]
    fn idct4x4_16_add_with_larger_stride() {
        let mut input = [0i32; 16];
        input[0] = 1024;
        // 8-wide dest, only first 4 columns affected per row.
        let mut dest = vec![10u8; 32]; // 4 rows × 8 stride
        idct4x4_16_add(&input, &mut dest, 8);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(dest[r * 8 + c], 42, "row {r} col {c} expected 10+32=42");
            }
            // Cols 4..8 untouched.
            for c in 4..8 {
                assert_eq!(dest[r * 8 + c], 10, "row {r} col {c} should be unchanged");
            }
        }
    }

    // ---- 8x8 family ----

    /// Zero input → zero output (1D).
    #[test]
    fn idct8_zero_in_zero_out() {
        let input = [0i32; 8];
        let mut output = [99i32; 8];
        idct8(&input, &mut output);
        assert_eq!(output, [0; 8]);
    }

    /// Zero input → unchanged dest. `(0 + 16) >> 5 = 0`.
    #[test]
    fn idct8x8_64_add_zero_preserves_dest() {
        let input = [0i32; 64];
        let mut dest = [128u8; 64];
        idct8x8_64_add(&input, &mut dest, 8);
        assert_eq!(dest, [128; 64]);
    }

    /// DC-only fast path must agree with full path when only `input[0]`
    /// is non-zero.
    #[test]
    fn idct8x8_1_add_matches_64_add_for_dc_only() {
        let dc: i32 = 256;
        let mut input = [0i32; 64];
        input[0] = dc;
        let mut dest_full = [50u8; 64];
        let mut dest_dc = [50u8; 64];
        idct8x8_64_add(&input, &mut dest_full, 8);
        idct8x8_1_add(dc, &mut dest_dc, 8);
        assert_eq!(
            dest_full, dest_dc,
            "DC-only fast path must match full path for DC-only input"
        );
    }

    /// Differential against a hand-computed reference for the DC-only
    /// path.
    ///
    /// Input is `[1024, 0, 0, ..., 0]`. Trace:
    /// - First cospi_16_64 pass: round_shift(1024 * 11585)
    ///   = round_shift(11862784) = (11862784 + 8192) >> 14
    ///   = 11870976 >> 14 = 724.
    /// - Second cospi_16_64 pass: round_shift(724 * 11585)
    ///   = round_shift(8387540) = (8387540 + 8192) >> 14
    ///   = 8395732 >> 14 = 512.
    /// - Final shift: (512 + 16) >> 5 = 528 >> 5 = 16.
    /// - Add 16 to each pixel of dest=0 → 16 everywhere.
    #[test]
    fn idct8x8_1_add_dc_known_value() {
        let mut dest = [0u8; 64];
        idct8x8_1_add(1024, &mut dest, 8);
        assert_eq!(dest, [16u8; 64]);
    }

    /// DC saturation: large positive DC clips at 255.
    #[test]
    fn idct8x8_1_add_saturates_high() {
        let mut dest = [200u8; 64];
        idct8x8_1_add(30_000, &mut dest, 8);
        assert_eq!(dest, [255; 64]);
    }

    /// DC saturation: large negative DC clips at 0.
    #[test]
    fn idct8x8_1_add_saturates_low() {
        let mut dest = [100u8; 64];
        idct8x8_1_add(-30_000, &mut dest, 8);
        assert_eq!(dest, [0; 64]);
    }

    /// idct8 with non-DC input: input `[0, 1024, 0, 0, 0, 0, 0, 0]`
    /// (one AC coefficient). The output should be a non-flat 8-vector
    /// that, when butterflied with itself in the 2D pass, produces
    /// pixel-correct residual. Sanity-test: output is non-zero.
    #[test]
    fn idct8_ac_input_produces_nonzero_output() {
        let mut input = [0i32; 8];
        input[1] = 1024;
        let mut output = [0i32; 8];
        idct8(&input, &mut output);
        assert!(output.iter().any(|&v| v != 0), "AC input → all-zero output");
    }

    /// 8×8 full path with a single AC coefficient: dest changes
    /// non-uniformly (would be uniform for DC, varied for AC).
    #[test]
    fn idct8x8_64_add_ac_modulates_pixels() {
        let mut input = [0i32; 64];
        input[1] = 2048; // AC coefficient at position (0, 1) — horizontal cosine
        let mut dest = [128u8; 64];
        idct8x8_64_add(&input, &mut dest, 8);
        // Some pixels increased, some decreased relative to 128.
        let any_above = dest.iter().any(|&v| v > 128);
        let any_below = dest.iter().any(|&v| v < 128);
        assert!(any_above && any_below, "AC should produce both signs");
    }

    /// Regression: `wraplow` (i16 modular truncation) is applied at
    /// every libvpx-`int16_t step[]` storage site. For an input whose
    /// post-`dct_const_round_shift` value crosses the i16 boundary, the
    /// output must match the wrapped value, NOT the untruncated value.
    ///
    /// Construction: pick `input[0]` such that
    /// `round_shift(input[0] * 11585)` equals exactly `i16::MAX + 1`
    /// (= 32768) post-rounding. That value, stored into `int16_t step`,
    /// wraps to `-32768`. Across the second cospi-16 pass and the >>4
    /// final shift, the wrapped output is observably different from
    /// what an i32-throughout implementation would produce.
    ///
    /// Working: `round_shift(x * 11585) = 32768` requires
    /// `(x * 11585 + 8192) >> 14 = 32768`, i.e. `x * 11585` in
    /// `[32768*16384 - 8192, 32768*16384 + 16383]`
    /// = `[536862720, 536887295]`. `x = 536862720 / 11585 ≈ 46341.18`,
    /// so `x = 46341` gives `46341 * 11585 = 536860485` → round_shift
    /// = `(536860485 + 8192) >> 14 = 536868677 >> 14 = 32767` (i16
    /// max, no wrap). `x = 46342` gives `46342 * 11585 = 536872070` →
    /// `(536872070 + 8192) >> 14 = 536880262 >> 14 = 32768` → wraps to
    /// `-32768` under i16 truncation.
    ///
    /// With wrapped `out1 = -32768`:
    /// `out2 = round_shift(-32768 * 11585) = (-379617280 + 8192) >> 14
    ///       = -379609088 >> 14 = -23170`. Fits in i16, no further wrap.
    /// `a1 = (-23170 + 8) >> 4 = -23162 >> 4 = -1448`. Add to dest=200:
    /// `200 + (-1448) = -1248`, clipped to 0.
    ///
    /// Without wrap (the old i32-storage implementation) `out1 = 32768`,
    /// `out2 = 23170`, `a1 = 1448`, dest = clip(200+1448) = 255.
    ///
    /// This test pins us to the wrapped (libvpx-bit-exact) output.
    #[test]
    fn idct4x4_1_add_wraplow_truncation_boundary() {
        let mut dest = [200u8; 16];
        idct4x4_1_add(46342, &mut dest, 4);
        assert_eq!(
            dest, [0u8; 16],
            "wraplow truncation at i16 boundary should sign-flip the residual"
        );
    }

    // ---- 16x16 family ----

    /// Zero input → zero output (1D).
    #[test]
    fn idct16_zero_in_zero_out() {
        let input = [0i32; 16];
        let mut output = [99i32; 16];
        idct16(&input, &mut output);
        assert_eq!(output, [0; 16]);
    }

    /// Zero input → unchanged dest. `(0 + 32) >> 6 = 0`.
    #[test]
    fn idct16x16_256_add_zero_preserves_dest() {
        let input = [0i32; 256];
        let mut dest = [128u8; 256];
        idct16x16_256_add(&input, &mut dest, 16);
        assert_eq!(dest, [128; 256]);
    }

    /// DC-only fast path must match full path when only `input[0]` is set.
    #[test]
    fn idct16x16_1_add_matches_256_add_for_dc_only() {
        let dc: i32 = 512;
        let mut input = [0i32; 256];
        input[0] = dc;
        let mut dest_full = [50u8; 256];
        let mut dest_dc = [50u8; 256];
        idct16x16_256_add(&input, &mut dest_full, 16);
        idct16x16_1_add(dc, &mut dest_dc, 16);
        assert_eq!(
            dest_full, dest_dc,
            "DC-only fast path must match full path for DC-only input"
        );
    }

    /// Hand-computed DC differential for `idct16x16_1_add(1024)`.
    ///
    /// Trace (with `wraplow`, no truncation observed for these
    /// magnitudes):
    /// - `out1 = round_shift(1024 * 11585) = round_shift(11862784)
    ///         = (11862784 + 8192) >> 14 = 11870976 >> 14 = 724`.
    /// - `out2 = round_shift(724 * 11585) = round_shift(8387540)
    ///         = (8387540 + 8192) >> 14 = 8395732 >> 14 = 512`.
    /// - `a1 = (512 + 32) >> 6 = 544 >> 6 = 8`.
    /// - Add 8 to dest=0 everywhere → 8.
    #[test]
    fn idct16x16_1_add_dc_known_value() {
        let mut dest = [0u8; 256];
        idct16x16_1_add(1024, &mut dest, 16);
        assert_eq!(dest, [8u8; 256]);
    }

    /// DC saturation: large positive DC clips at 255.
    #[test]
    fn idct16x16_1_add_saturates_high() {
        let mut dest = [200u8; 256];
        idct16x16_1_add(20_000, &mut dest, 16);
        assert_eq!(dest, [255; 256]);
    }

    /// DC saturation: large negative DC clips at 0.
    #[test]
    fn idct16x16_1_add_saturates_low() {
        let mut dest = [100u8; 256];
        idct16x16_1_add(-20_000, &mut dest, 16);
        assert_eq!(dest, [0; 256]);
    }

    /// idct16 with single AC coefficient: output is non-zero and not
    /// flat (would be flat for DC).
    #[test]
    fn idct16_ac_input_produces_varied_output() {
        let mut input = [0i32; 16];
        input[1] = 1024;
        let mut output = [0i32; 16];
        idct16(&input, &mut output);
        let any_nonzero = output.iter().any(|&v| v != 0);
        let varied = output.iter().any(|&v| v != output[0]);
        assert!(any_nonzero && varied, "AC input → flat/zero output");
    }

    /// 16×16 full path with one AC coefficient produces both signs of
    /// residual.
    #[test]
    fn idct16x16_256_add_ac_modulates_pixels() {
        let mut input = [0i32; 256];
        input[1] = 4096;
        let mut dest = [128u8; 256];
        idct16x16_256_add(&input, &mut dest, 16);
        let any_above = dest.iter().any(|&v| v > 128);
        let any_below = dest.iter().any(|&v| v < 128);
        assert!(any_above && any_below, "AC should produce both signs");
    }

    // ---- 32x32 family ----

    /// Zero input → zero output (1D).
    #[test]
    fn idct32_zero_in_zero_out() {
        let input = [0i32; 32];
        let mut output = [99i32; 32];
        idct32(&input, &mut output);
        assert_eq!(output, [0; 32]);
    }

    /// Zero input → unchanged dest. `(0 + 32) >> 6 = 0`.
    #[test]
    fn idct32x32_1024_add_zero_preserves_dest() {
        let input = [0i32; 1024];
        let mut dest = vec![128u8; 1024];
        idct32x32_1024_add(&input, &mut dest, 32);
        assert!(dest.iter().all(|&v| v == 128));
    }

    /// DC-only fast path must match full path when only `input[0]` is set.
    #[test]
    fn idct32x32_1_add_matches_1024_add_for_dc_only() {
        let dc: i32 = 256;
        let mut input = [0i32; 1024];
        input[0] = dc;
        let mut dest_full = vec![50u8; 1024];
        let mut dest_dc = vec![50u8; 1024];
        idct32x32_1024_add(&input, &mut dest_full, 32);
        idct32x32_1_add(dc, &mut dest_dc, 32);
        assert_eq!(
            dest_full, dest_dc,
            "DC-only fast path must match full path for DC-only input"
        );
    }

    /// Hand-computed DC differential for `idct32x32_1_add(1024)`.
    ///
    /// Trace (with `wraplow`, no truncation observed for these
    /// magnitudes):
    /// - `out1 = round_shift(1024 * 11585) = 724`.
    /// - `out2 = round_shift(724 * 11585) = 512`.
    /// - `a1 = (512 + 32) >> 6 = 544 >> 6 = 8`.
    /// - Add 8 to dest=0 everywhere → 8.
    ///
    /// Identical to the 16×16 DC case (same two cospi_16_64 passes
    /// and `>>6` shift).
    #[test]
    fn idct32x32_1_add_dc_known_value() {
        let mut dest = vec![0u8; 1024];
        idct32x32_1_add(1024, &mut dest, 32);
        assert!(dest.iter().all(|&v| v == 8));
    }

    /// DC saturation: large positive DC clips at 255.
    #[test]
    fn idct32x32_1_add_saturates_high() {
        let mut dest = vec![200u8; 1024];
        idct32x32_1_add(20_000, &mut dest, 32);
        assert!(dest.iter().all(|&v| v == 255));
    }

    /// DC saturation: large negative DC clips at 0.
    #[test]
    fn idct32x32_1_add_saturates_low() {
        let mut dest = vec![100u8; 1024];
        idct32x32_1_add(-20_000, &mut dest, 32);
        assert!(dest.iter().all(|&v| v == 0));
    }

    /// idct32 with single AC coefficient produces a varied 32-vector.
    #[test]
    fn idct32_ac_input_produces_varied_output() {
        let mut input = [0i32; 32];
        input[1] = 1024;
        let mut output = [0i32; 32];
        idct32(&input, &mut output);
        let any_nonzero = output.iter().any(|&v| v != 0);
        let varied = output.iter().any(|&v| v != output[0]);
        assert!(any_nonzero && varied, "AC input → flat/zero output");
    }

    /// 32×32 full path with one AC coefficient produces both signs of
    /// residual.
    #[test]
    fn idct32x32_1024_add_ac_modulates_pixels() {
        let mut input = [0i32; 1024];
        input[1] = 4096;
        let mut dest = vec![128u8; 1024];
        idct32x32_1024_add(&input, &mut dest, 32);
        let any_above = dest.iter().any(|&v| v > 128);
        let any_below = dest.iter().any(|&v| v < 128);
        assert!(any_above && any_below, "AC should produce both signs");
    }

    // ---- iadst family ----

    /// Zero input → zero output (4-point ADST early-out).
    #[test]
    fn iadst4_zero_in_zero_out() {
        let input = [0i32; 4];
        let mut output = [99i32; 4];
        iadst4(&input, &mut output);
        assert_eq!(output, [0; 4]);
    }

    /// iadst4 with `x = [1024, 0, 0, 0]`. Hand-computed trace.
    ///
    /// Inputs after libvpx's identity permutation: x0=1024, x1=x2=x3=0
    /// (non-zero — no early-out).
    ///
    /// Stage-1 products: `s0_a = 5283 * 1024 = 5_409_792`,
    /// `s1_a = 9929 * 1024 = 10_167_296`, all others zero. `s7 =
    /// wraplow(1024 - 0 + 0) = 1024`. Then `s0 = s0_a + s3_a + s5_a =
    /// 5_409_792`, `s1 = s1_a - s4_a - s6_a = 10_167_296`, `s3 = s2_a
    /// = 0`, `s2 = SINPI_3_9 * s7 = 13377 * 1024 = 13_698_048`.
    ///
    /// Output via `wraplow(round_shift(_))`:
    ///
    /// `output[0]`: `round_shift(5_409_792) = (5_409_792 + 8192) >> 14
    /// = 5_417_984 >> 14 = 330`.
    ///
    /// `output[1]`: `round_shift(10_167_296) = (10_167_296 + 8192) >> 14
    /// = 10_175_488 >> 14 = 621`.
    ///
    /// `output[2]`: `round_shift(13_698_048) = (13_698_048 + 8192) >> 14
    /// = 13_706_240 >> 14 = 836`.
    ///
    /// `output[3]`: `round_shift(5_409_792 + 10_167_296 - 0)
    /// = round_shift(15_577_088) = (15_577_088 + 8192) >> 14
    /// = 15_585_280 >> 14 = 951`.
    #[test]
    fn iadst4_x0_only_known_value() {
        let input = [1024_i32, 0, 0, 0];
        let mut output = [0i32; 4];
        iadst4(&input, &mut output);
        assert_eq!(output, [330, 621, 836, 951]);
    }

    /// Zero input → zero output (8-point ADST early-out).
    #[test]
    fn iadst8_zero_in_zero_out() {
        let input = [0i32; 8];
        let mut output = [99i32; 8];
        iadst8(&input, &mut output);
        assert_eq!(output, [0; 8]);
    }

    /// iadst8 single non-zero input produces a non-zero output vector.
    #[test]
    fn iadst8_nonzero_input_produces_nonzero_output() {
        let mut input = [0i32; 8];
        input[0] = 1024; // maps to x1 per libvpx input permutation
        let mut output = [0i32; 8];
        iadst8(&input, &mut output);
        assert!(
            output.iter().any(|&v| v != 0),
            "non-zero input should yield non-zero output"
        );
    }

    /// Zero input → zero output (16-point ADST early-out).
    #[test]
    fn iadst16_zero_in_zero_out() {
        let input = [0i32; 16];
        let mut output = [99i32; 16];
        iadst16(&input, &mut output);
        assert_eq!(output, [0; 16]);
    }

    /// iadst16 single non-zero input produces a non-zero output vector.
    #[test]
    fn iadst16_nonzero_input_produces_nonzero_output() {
        let mut input = [0i32; 16];
        input[0] = 2048;
        let mut output = [0i32; 16];
        iadst16(&input, &mut output);
        assert!(
            output.iter().any(|&v| v != 0),
            "non-zero input should yield non-zero output"
        );
    }

    // ---- iht (hybrid) family ----

    /// `iht4x4_16_add` with `DctDct` is bit-identical to
    /// `idct4x4_16_add` (DCT row, DCT col → same path).
    #[test]
    fn iht4x4_dct_dct_matches_idct4x4_16_add() {
        let mut input = [0i32; 16];
        input[0] = 1024;
        input[1] = 256;
        input[5] = -128;
        let mut dest_iht = [50u8; 16];
        let mut dest_idct = [50u8; 16];
        iht4x4_16_add(&input, &mut dest_iht, 4, TxType::DctDct);
        idct4x4_16_add(&input, &mut dest_idct, 4);
        assert_eq!(dest_iht, dest_idct, "DctDct must match pure idct 4x4");
    }

    /// `iht8x8_64_add` with `DctDct` matches `idct8x8_64_add`.
    #[test]
    fn iht8x8_dct_dct_matches_idct8x8_64_add() {
        let mut input = [0i32; 64];
        input[0] = 512;
        input[1] = 200;
        input[9] = -64;
        let mut dest_iht = [80u8; 64];
        let mut dest_idct = [80u8; 64];
        iht8x8_64_add(&input, &mut dest_iht, 8, TxType::DctDct);
        idct8x8_64_add(&input, &mut dest_idct, 8);
        assert_eq!(dest_iht, dest_idct, "DctDct must match pure idct 8x8");
    }

    /// `iht16x16_256_add` with `DctDct` matches `idct16x16_256_add`.
    #[test]
    fn iht16x16_dct_dct_matches_idct16x16_256_add() {
        let mut input = [0i32; 256];
        input[0] = 1024;
        input[1] = 128;
        input[17] = -32;
        let mut dest_iht = vec![100u8; 256];
        let mut dest_idct = vec![100u8; 256];
        iht16x16_256_add(&input, &mut dest_iht, 16, TxType::DctDct);
        idct16x16_256_add(&input, &mut dest_idct, 16);
        assert_eq!(dest_iht, dest_idct, "DctDct must match pure idct 16x16");
    }

    /// `iht4x4_16_add` with all-zero input preserves `dest` for every
    /// TxType.
    #[test]
    fn iht4x4_zero_preserves_dest_all_tx_types() {
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let input = [0i32; 16];
            let mut dest = [128u8; 16];
            iht4x4_16_add(&input, &mut dest, 4, tx_type);
            assert_eq!(dest, [128; 16], "tx_type {tx_type:?} did not preserve zero-input dest");
        }
    }

    /// `iht8x8_64_add` with all-zero input preserves `dest` for every
    /// TxType.
    #[test]
    fn iht8x8_zero_preserves_dest_all_tx_types() {
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let input = [0i32; 64];
            let mut dest = [128u8; 64];
            iht8x8_64_add(&input, &mut dest, 8, tx_type);
            assert_eq!(dest, [128; 64], "tx_type {tx_type:?} did not preserve zero-input dest");
        }
    }

    /// `iht16x16_256_add` with all-zero input preserves `dest` for
    /// every TxType.
    #[test]
    fn iht16x16_zero_preserves_dest_all_tx_types() {
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let input = [0i32; 256];
            let mut dest = vec![128u8; 256];
            iht16x16_256_add(&input, &mut dest, 16, tx_type);
            assert!(
                dest.iter().all(|&v| v == 128),
                "tx_type {tx_type:?} did not preserve zero-input dest"
            );
        }
    }

    /// `iht4x4_16_add(DctAdst, ...)` with `dq_coefs = [64, 0, 0, ..., 0]`.
    /// Hand-traced oracle test for M.2 bisect of the H-mode (DCT_ADST)
    /// residual divergence — UPDATED post Stage 3 stretch M.8.
    ///
    /// libvpx's `IHT_4[DCT_ADST] = { idct4_c, iadst4_c }` is
    /// `{cols, rows}` order, so for DctAdst: rows=iadst4, cols=idct4.
    ///
    /// Trace (row pass iadst4 on [64, 0, 0, 0]):
    ///   - s0_a = 5283 * 64 = 338_112
    ///   - s1_a = 9929 * 64 = 635_456
    ///   - s7 = 64 ; s2 = 13377 * 64 = 856_128
    ///   - s0 = 338_112; s1 = 635_456; s3 = 0
    ///   - out[0] = (338_112 + 8192) >> 14 = 21
    ///   - out[1] = (635_456 + 8192) >> 14 = 39
    ///   - out[2] = (856_128 + 8192) >> 14 = 52
    ///   - out[3] = (338_112 + 635_456 + 8192) >> 14 = 59
    /// - intermediate row 0 = [21, 39, 52, 59]; rows 1..3 = 0.
    /// - Column pass idct4 on col 0 = [21, 0, 0, 0] gives [15, 15, 15, 15].
    /// - Column 1 idct4([39, 0, 0, 0]) gives [28; 4]. Column 2 [37; 4]. Column 3 [42; 4].
    /// - ROUND_POWER_OF_TWO each: col0=1, col1=2, col2=2, col3=3.
    /// - Each row gets [1, 2, 2, 3].
    #[test]
    fn iht4x4_dct_adst_dc_only_known_value() {
        let mut input = [0i32; 16];
        input[0] = 64;
        let mut dest = [0u8; 16];
        iht4x4_16_add(&input, &mut dest, 4, TxType::DctAdst);
        let row = [1u8, 2, 2, 3];
        let expected: [u8; 16] = [
            row[0], row[1], row[2], row[3],
            row[0], row[1], row[2], row[3],
            row[0], row[1], row[2], row[3],
            row[0], row[1], row[2], row[3],
        ];
        assert_eq!(dest, expected);
    }

    /// `iht4x4_16_add(AdstDct, ...)` with `dq_coefs = [64, 0, 0, ..., 0]`.
    /// Hand-traced oracle test — UPDATED post Stage 3 stretch M.8.
    ///
    /// libvpx's `IHT_4[ADST_DCT] = { iadst4_c, idct4_c }` is
    /// `{cols, rows}` order, so for AdstDct: rows=idct4, cols=iadst4.
    ///
    /// Trace (row pass idct4 on [64, 0, 0, 0]):
    ///   - temp1 = (64+0) * 11585 = 741_440 → (741_440+8192) >> 14 = 45
    ///   - step0 = step1 = 45, step2 = step3 = 0
    ///   - output = [45, 45, 45, 45]
    /// - intermediate row 0 = [45, 45, 45, 45]; rows 1..3 = [0; 4].
    /// - Column pass iadst4 on each col = [45, 0, 0, 0] gives [15, 27, 37, 42].
    /// - ROUND_POWER_OF_TWO each: row 0=1, row 1=2, row 2=2, row 3=3.
    /// - Each row is filled identically across columns.
    #[test]
    fn iht4x4_adst_dct_dc_only_known_value() {
        let mut input = [0i32; 16];
        input[0] = 64;
        let mut dest = [0u8; 16];
        iht4x4_16_add(&input, &mut dest, 4, TxType::AdstDct);
        let expected: [u8; 16] = [
            1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
        ];
        assert_eq!(dest, expected);
    }

    /// The three non-DCT-DCT TxTypes produce different output from
    /// DctDct for the same non-trivial input (sanity check that the
    /// dispatch actually changes behaviour).
    #[test]
    fn iht4x4_adst_variants_diverge_from_dct_dct() {
        let mut input = [0i32; 16];
        input[0] = 1024;
        input[5] = 256;
        let mut base = [50u8; 16];
        iht4x4_16_add(&input, &mut base, 4, TxType::DctDct);
        for tx_type in [TxType::AdstDct, TxType::DctAdst, TxType::AdstAdst] {
            let mut other = [50u8; 16];
            iht4x4_16_add(&input, &mut other, 4, tx_type);
            assert_ne!(
                base, other,
                "tx_type {tx_type:?} should diverge from DctDct"
            );
        }
    }

    // -----------------------------------------------------------------
    // Phase-1 differential oracle — Stage 3 stretch M.3
    //
    // Goal: confirm that the production `iht4x4_16_add` matches a
    // bit-for-bit independently-transcribed port of libvpx's
    // `vp9_iht4x4_16_add_c` on multi-coef inputs. M.2's tests already
    // covered DC-only inputs; this fills in the multi-coef gap.
    //
    // The reference port deliberately re-implements `iadst4`, `idct4`
    // and the 16-iter driver from libvpx C source (`vp9_idct.c:20` +
    // `vpx_dsp/inv_txfm.c:96, 133`) using primitive ops only — no
    // shared helpers with the production code (`wraplow`,
    // `dct_const_round_shift`, `clip_pixel_add` are all re-inlined).
    //
    // If both sides agree across 100 randomised inputs, we can
    // localise the cascade bug elsewhere (dq_coefs production).
    // -----------------------------------------------------------------

    /// `sinpi_*_9` constants — libvpx `vpx_dsp/txfm_common.h:61-64`.
    const REF_SINPI_1_9: i64 = 5283;
    const REF_SINPI_2_9: i64 = 9929;
    const REF_SINPI_3_9: i64 = 13377;
    const REF_SINPI_4_9: i64 = 15212;

    /// `cospi_*_64` constants — libvpx `vpx_dsp/txfm_common.h:35-51`.
    const REF_COSPI_8_64: i64 = 15137;
    const REF_COSPI_16_64: i64 = 11585;
    const REF_COSPI_24_64: i64 = 6270;

    /// `DCT_CONST_BITS = 14`. Reference port of
    /// `dct_const_round_shift(x) = ROUND_POWER_OF_TWO(x, 14)`
    /// (`vpx_dsp/inv_txfm.h:38-41`).
    fn ref_dct_const_round_shift(x: i64) -> i64 {
        // ROUND_POWER_OF_TWO(value, n) is `((value) + (((1 << (n)) >> 1))) >> (n)`.
        // We're operating on signed arithmetic where the C type is
        // `tran_high_t` (int64_t).
        (x + (1 << 13)) >> 14
    }

    /// `WRAPLOW(x)` non-emulate-hardware branch — `(int32_t)x`. Our
    /// inputs are already i32, so this is a no-op cast.
    /// (`vpx_dsp/inv_txfm.h:87`).
    fn ref_wraplow(x: i64) -> i32 {
        x as i32
    }

    /// `clip_pixel_add(dest, trans)` — `vpx_dsp/inv_txfm.h:117-120`.
    fn ref_clip_pixel_add(dest: u8, trans: i64) -> u8 {
        let trans = ref_wraplow(trans) as i64;
        let v = dest as i64 + trans;
        if v < 0 {
            0
        } else if v > 255 {
            255
        } else {
            v as u8
        }
    }

    /// Reference `iadst4_c` — `vpx_dsp/inv_txfm.c:96-131`.
    fn ref_iadst4(input: &[i32; 4], output: &mut [i32; 4]) {
        let x0 = input[0] as i64;
        let x1 = input[1] as i64;
        let x2 = input[2] as i64;
        let x3 = input[3] as i64;

        if (x0 | x1 | x2 | x3) == 0 {
            *output = [0; 4];
            return;
        }

        let s0 = REF_SINPI_1_9 * x0;
        let s1 = REF_SINPI_2_9 * x0;
        let s2 = REF_SINPI_3_9 * x1;
        let s3 = REF_SINPI_4_9 * x2;
        let s4 = REF_SINPI_1_9 * x2;
        let s5 = REF_SINPI_2_9 * x3;
        let s6 = REF_SINPI_4_9 * x3;
        // libvpx: `s7 = WRAPLOW(x0 - x2 + x3);` — i32 truncation point.
        let s7 = ref_wraplow(x0 - x2 + x3) as i64;

        let s0p = s0 + s3 + s5;
        let s1p = s1 - s4 - s6;
        let s3p = s2;
        let s2p = REF_SINPI_3_9 * s7;

        output[0] = ref_wraplow(ref_dct_const_round_shift(s0p + s3p));
        output[1] = ref_wraplow(ref_dct_const_round_shift(s1p + s3p));
        output[2] = ref_wraplow(ref_dct_const_round_shift(s2p));
        output[3] = ref_wraplow(ref_dct_const_round_shift(s0p + s1p - s3p));
    }

    /// Reference `idct4_c` — `vpx_dsp/inv_txfm.c:133-152`.
    /// libvpx stores `step[]` as `int16_t`; our reference uses an
    /// explicit `as i16 as i32` round-trip at that point to mirror
    /// the implicit truncation. The input is already i16-range so the
    /// `(int16_t)input[i]` reads in the C source are no-ops.
    fn ref_idct4(input: &[i32; 4], output: &mut [i32; 4]) {
        let temp1 = (input[0] as i64 + input[2] as i64) * REF_COSPI_16_64;
        let temp2 = (input[0] as i64 - input[2] as i64) * REF_COSPI_16_64;
        let step0 = ref_wraplow(ref_dct_const_round_shift(temp1)) as i16 as i32;
        let step1 = ref_wraplow(ref_dct_const_round_shift(temp2)) as i16 as i32;
        let temp1 =
            input[1] as i64 * REF_COSPI_24_64 - input[3] as i64 * REF_COSPI_8_64;
        let temp2 =
            input[1] as i64 * REF_COSPI_8_64 + input[3] as i64 * REF_COSPI_24_64;
        let step2 = ref_wraplow(ref_dct_const_round_shift(temp1)) as i16 as i32;
        let step3 = ref_wraplow(ref_dct_const_round_shift(temp2)) as i16 as i32;

        output[0] = ref_wraplow(step0 as i64 + step3 as i64);
        output[1] = ref_wraplow(step1 as i64 + step2 as i64);
        output[2] = ref_wraplow(step1 as i64 - step2 as i64);
        output[3] = ref_wraplow(step0 as i64 - step3 as i64);
    }

    /// Reference `vp9_iht4x4_16_add_c` — `vp9/common/vp9_idct.c:20-50`.
    fn ref_iht4x4_16_add(
        input: &[i32; 16],
        dest: &mut [u8],
        stride: usize,
        tx_type: TxType,
    ) {
        type Xform = fn(&[i32; 4], &mut [i32; 4]);
        // Stage 3 stretch M.8 fix: libvpx's `transform_2d` struct is
        // `{ transform_1d cols, rows; }`. Brace-initializer order is
        // `{cols, rows}`, so `IHT_4[ADST_DCT] = {iadst4_c, idct4_c}`
        // means cols=iadst4, rows=idct4 — i.e. row-pass = idct4,
        // col-pass = iadst4.
        let (rows, cols): (Xform, Xform) = match tx_type {
            TxType::DctDct => (ref_idct4, ref_idct4),
            TxType::AdstDct => (ref_idct4, ref_iadst4),
            TxType::DctAdst => (ref_iadst4, ref_idct4),
            TxType::AdstAdst => (ref_iadst4, ref_iadst4),
        };

        let mut out = [0i32; 16];

        // Row pass.
        for i in 0..4 {
            let row_in: &[i32; 4] = input[i * 4..i * 4 + 4].try_into().unwrap();
            let row_out: &mut [i32; 4] =
                (&mut out[i * 4..i * 4 + 4]).try_into().unwrap();
            rows(row_in, row_out);
        }

        // Col pass + add. libvpx: `ROUND_POWER_OF_TWO(temp_out[j], 4)`.
        for i in 0..4 {
            let mut temp_in = [0i32; 4];
            for j in 0..4 {
                temp_in[j] = out[j * 4 + i];
            }
            let mut temp_out = [0i32; 4];
            cols(&temp_in, &mut temp_out);
            for j in 0..4 {
                let rounded = (temp_out[j] as i64 + 8) >> 4;
                dest[j * stride + i] =
                    ref_clip_pixel_add(dest[j * stride + i], rounded);
            }
        }
    }

    /// Deterministic LCG — keeps the test hermetic so failures
    /// reproduce. Same seed every run.
    fn lcg_next(state: &mut u64) -> u64 {
        // Numerical Recipes parameters.
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    /// N=100 random `[i32; 16]` inputs in the realistic dequant range
    /// `[-256, +256]`. Differential test the production
    /// `iht4x4_16_add` against the bit-for-bit `ref_iht4x4_16_add`
    /// reference port for every TxType. Pixel-exact match required.
    #[test]
    fn iht4x4_16_add_matches_libvpx_reference_random_inputs() {
        const N: usize = 100;
        const STRIDE: usize = 4;
        let mut state: u64 = 0xC0DE_FACE_DEAD_BEEF;

        for sample in 0..N {
            // Realistic dequant range: small AC coefficients dominate.
            // Mix in a wider range every ~5 samples to stress edges.
            let range: i32 = if sample % 5 == 0 { 1024 } else { 256 };
            let mut input = [0i32; 16];
            for slot in input.iter_mut() {
                let r = lcg_next(&mut state) as i32;
                // Map u32→i32 in `[-range, range]`.
                let signed = ((r as i64 % (2 * range as i64 + 1)) - range as i64) as i32;
                *slot = signed;
            }
            // Pre-existing pixel content (non-zero) to catch any
            // saturation / clip_pixel_add divergence as well as the
            // residual numerics. Use a deterministic pattern.
            let mut dest_seed = [0u8; 16];
            for (k, slot) in dest_seed.iter_mut().enumerate() {
                *slot = ((sample.wrapping_mul(7) + k * 13) & 0xFF) as u8;
            }

            for tx_type in [
                TxType::DctDct,
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
            ] {
                let mut dest_prod = dest_seed;
                let mut dest_ref = dest_seed;
                iht4x4_16_add(&input, &mut dest_prod, STRIDE, tx_type);
                ref_iht4x4_16_add(&input, &mut dest_ref, STRIDE, tx_type);
                assert_eq!(
                    dest_prod, dest_ref,
                    "iht4x4_16_add divergence at sample {sample}, tx_type {tx_type:?}\n\
                     input = {input:?}\n\
                     dest_seed = {dest_seed:?}\n\
                     prod = {dest_prod:?}\n\
                     ref  = {dest_ref:?}"
                );
            }
        }
    }

    /// Targeted probe at the H-mode cascade pattern reported by M.1 /
    /// M.2: TxType::DctAdst (H-mode → col_scan), small AC coefficients
    /// such as might emerge from a real 4×4 H-mode residual. This is
    /// the realistic failure shape, not a synthetic stress test.
    #[test]
    fn iht4x4_dct_adst_h_mode_pattern_matches_reference() {
        // Hand-picked inputs that probe the iadst4 (column pass) on
        // patterns with non-trivial dq_coefs across multiple
        // scan-positions. Includes the cascade-suspect pattern.
        let inputs: &[[i32; 16]] = &[
            // DC + a few small AC. Realistic H-mode 4×4 residual shape.
            [16, -8, 4, 0,
              0,  0, 0, 0,
              0,  0, 0, 0,
              0,  0, 0, 0],
            // A negative DC + scan-position-1 col_scan pattern.
            [-32, 16, 0, 0,
              -8,  4, 0, 0,
              0,   0, 0, 0,
              0,   0, 0, 0],
            // The kind of coef distribution that produces the
            // M.1-reported residual `[+0f, +09, -10, +03]` order
            // of magnitude.
            [64, -16, 8, -4,
              0,   0, 0,  0,
              0,   0, 0,  0,
              0,   0, 0,  0],
        ];

        for (idx, input) in inputs.iter().enumerate() {
            let mut dst_prod = [128u8; 16];
            let mut dst_ref = [128u8; 16];
            iht4x4_16_add(input, &mut dst_prod, 4, TxType::DctAdst);
            ref_iht4x4_16_add(input, &mut dst_ref, 4, TxType::DctAdst);
            assert_eq!(
                dst_prod, dst_ref,
                "H-mode pattern {idx}: prod != ref\n\
                 input = {input:?}\nprod = {dst_prod:?}\nref = {dst_ref:?}"
            );
        }
    }

    /// Stage 3 stretch M.16 — captured dq_coefs from resize frame 1 at
    /// the Block8x8 Dc block at (px 280, 0): `dc=-698, ac@(0,1)=550`,
    /// dest pre-filled with Dc pred = 69. Confirms `idct8x8_64_add`
    /// produces a flat row 0..7 result of `[70, 68, 65, 60, 56, 51,
    /// 48, 46]` for this sparse input — matching our actual pre-LF
    /// frame output but NOT libvpx's pre-LF (col 7 varies in libvpx,
    /// pointing at a decode_coefs divergence upstream that yields
    /// different non-zero coefs into iht).
    #[test]
    fn m16_idct8x8_resize_280_0_probe() {
        let mut input = [0i32; 64];
        input[0] = -698;
        input[1] = 550;
        let mut dest = [69u8; 64];
        idct8x8_64_add(&input, &mut dest, 8);
        let expected_row: [u8; 8] = [70, 68, 65, 60, 56, 51, 48, 46];
        for r in 0..8 {
            assert_eq!(
                &dest[r * 8..r * 8 + 8],
                &expected_row,
                "row {r}: idct8x8 output differs from expected (M.16 pin)"
            );
        }
    }

    // ----- Walsh-Hadamard inverse (lossless) -------------------------
    //
    // G2 (decoder-hardening V6 §G2): scalar reference port of libvpx's
    // `vpx_iwht4x4_16_add_c` / `vpx_iwht4x4_1_add_c`
    // (`vpx_dsp/inv_txfm.c:18-94`). Each pass:
    //
    //   a1 = ip[0] >> UNIT_QUANT_SHIFT  (>>2 in default build)
    //   c1 = ip[1] >> UNIT_QUANT_SHIFT
    //   d1 = ip[2] >> UNIT_QUANT_SHIFT
    //   b1 = ip[3] >> UNIT_QUANT_SHIFT
    //   a1 += c1
    //   d1 -= b1
    //   e1 = (a1 - d1) >> 1
    //   b1 = e1 - b1
    //   c1 = e1 - c1
    //   a1 -= b1
    //   d1 += c1
    //   write {a1, b1, c1, d1} via WRAPLOW
    //
    // Tests below derive expected outputs by walking that recurrence
    // by hand, then cross-check the DC fast path against the 16-add
    // path for DC-only inputs.

    /// Zero input → unchanged dest.
    #[test]
    fn iwht4x4_16_add_zero_preserves_dest() {
        let input = [0i32; 16];
        let mut dest = [128u8; 16];
        iwht4x4_16_add(&input, &mut dest, 4);
        assert_eq!(dest, [128u8; 16]);
    }

    /// DC=16 single-coefficient input. Trace through the WHT:
    /// - Row pass on row 0 (`ip = [16, 0, 0, 0]`):
    ///   `a1 = 16 >> 2 = 4`, `c1 = d1 = b1 = 0`.
    ///   `a1 += c1 → 4`, `d1 -= b1 → 0`,
    ///   `e1 = (4 - 0) >> 1 = 2`,
    ///   `b1 = 2 - 0 = 2`, `c1 = 2 - 0 = 2`,
    ///   `a1 -= b1 → 2`, `d1 += c1 → 2`.
    ///   Row 0 output = `[2, 2, 2, 2]`. Rows 1..3 stay zero.
    /// - Column pass on col 0 (`ip = [2, 0, 0, 0]`): identical algebra
    ///   gives `[1, 1, 1, 1]`. By symmetry every column is `[1; 4]`.
    /// - `clip_pixel_add` with `trans = 1` and `dest = 128` → 129.
    #[test]
    fn iwht4x4_16_add_dc_16_known_value() {
        let mut input = [0i32; 16];
        input[0] = 16;
        let mut dest = [128u8; 16];
        iwht4x4_16_add(&input, &mut dest, 4);
        assert_eq!(dest, [129u8; 16]);
    }

    /// DC=64 single-coefficient input. Same trace as the DC=16 test
    /// but with `a1 = 64 >> 2 = 16`; the recurrence yields row-pass
    /// output `[8, 8, 8, 8]`, column-pass output `[4, 4, 4, 4]`, and
    /// final `dest = [132; 16]` from a 128 base.
    #[test]
    fn iwht4x4_16_add_dc_64_known_value() {
        let mut input = [0i32; 16];
        input[0] = 64;
        let mut dest = [128u8; 16];
        iwht4x4_16_add(&input, &mut dest, 4);
        assert_eq!(dest, [132u8; 16]);
    }

    /// `iwht4x4_1_add` is the DC-only fast path. For any DC-only input
    /// (`input[0] = dc`, rest zero), it must produce the exact same
    /// pixels as the full `iwht4x4_16_add`. Mirrors the equivalent
    /// `idct4x4_1_add_matches_16_add_for_dc_only` invariant.
    ///
    /// Inputs include both multiples of 4 (clean `>> UNIT_QUANT_SHIFT`)
    /// and non-multiples (`±5, ±7, ±25`) to exercise the arithmetic
    /// right-shift behavior of `>> UNIT_QUANT_SHIFT` for inputs that do
    /// not divide evenly — both paths must round identically.
    #[test]
    fn iwht4x4_1_add_matches_16_add_for_dc_only() {
        for dc in [
            -256i32, -64, -25, -16, -7, -5, -4, 0, 4, 5, 7, 16, 25, 64, 256,
        ] {
            let mut input = [0i32; 16];
            input[0] = dc;
            let mut dest_full = [128u8; 16];
            let mut dest_dc = [128u8; 16];
            iwht4x4_16_add(&input, &mut dest_full, 4);
            iwht4x4_1_add(dc, &mut dest_dc, 4);
            assert_eq!(
                dest_full, dest_dc,
                "DC={dc}: WHT DC-only fast path must match full WHT for DC-only input"
            );
        }
    }

    /// `clip_pixel_add` clamps to 255: a large positive DC produces a
    /// per-pixel residual well above what `dest = 200` can absorb, so the
    /// output saturates at the upper bound. Note: this does NOT exercise
    /// wraplow's i16 truncation — DC=16000 yields intermediate values
    /// around ~4000, well inside i16 range. It exclusively tests the
    /// `[0, 255]` clamp inside `clip_pixel_add`.
    #[test]
    fn iwht4x4_16_add_clamps_dest_to_255_on_high_dc() {
        let mut input = [0i32; 16];
        input[0] = 16_000;
        let mut dest = [200u8; 16];
        iwht4x4_16_add(&input, &mut dest, 4);
        assert_eq!(dest, [255u8; 16]);
    }

    /// Saturation: large negative DC clamps to 0.
    #[test]
    fn iwht4x4_16_add_saturates_low() {
        let mut input = [0i32; 16];
        input[0] = -16_000;
        let mut dest = [100u8; 16];
        iwht4x4_16_add(&input, &mut dest, 4);
        assert_eq!(dest, [0u8; 16]);
    }

    /// AC-only differential trace. Input has a single non-zero in row 0
    /// at position 1 (`c1`-slot of the row pass), so the row output
    /// involves all four lanes of the recurrence, not just the DC
    /// pattern. Cross-check the WHT output by walking the recurrence
    /// by hand on the resulting 4×4 intermediate.
    ///
    /// Input: `coef[1] = 32`. UNIT_QUANT_SHIFT divides by 4 → c1=8.
    /// - Row 0 (`ip = [0, 32, 0, 0]`): `a1=0, c1=8, d1=0, b1=0`,
    ///   `a1 += c1 → 8`, `d1 -= b1 → 0`, `e1 = (8-0)>>1 = 4`,
    ///   `b1 = 4-0 = 4`, `c1 = 4-8 = -4`, `a1 -= b1 → 4`, `d1 += c1 → -4`.
    ///   Row 0 output = `[4, 4, -4, -4]`.
    /// - Column 0 (`ip = [4, 0, 0, 0]`): `a1=4, c1=d1=b1=0` →
    ///   `e1 = 4>>1 = 2`, `b1 = 2`, `c1 = 2`, `a1 = 2`, `d1 = 2`.
    ///   Column 0 final pixel adds = `[2, 2, 2, 2]`.
    /// - Column 1 (`ip = [4, 0, 0, 0]`): same → `[2, 2, 2, 2]`.
    /// - Column 2 (`ip = [-4, 0, 0, 0]`): `a1=-4, e1=(-4)>>1 = -2`,
    ///   `b1 = -2`, `c1 = -2`, `a1 = -4-(-2) = -2`, `d1 = 0+(-2) = -2`.
    ///   Column 2 final pixel adds = `[-2, -2, -2, -2]`.
    /// - Column 3: same as column 2 → `[-2, -2, -2, -2]`.
    ///
    /// `dest = 128 + [2, 2, -2, -2]` for each row.
    #[test]
    fn iwht4x4_16_add_ac_row0_pos1_known_value() {
        let mut input = [0i32; 16];
        input[1] = 32;
        let mut dest = [128u8; 16];
        iwht4x4_16_add(&input, &mut dest, 4);
        let row: [u8; 4] = [130, 130, 126, 126];
        for r in 0..4 {
            assert_eq!(&dest[r * 4..r * 4 + 4], &row, "row {r}");
        }
    }
}
