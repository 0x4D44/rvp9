//! Inverse transform scalar reference kernels (decoder + encoder
//! reconstruction).
//!
//! Currently lands the 4×4 family only:
//!
//! - [`idct4`] — 1D 4-point inverse DCT. Mirrors libvpx `idct4_c`
//!   (`vpx_dsp/inv_txfm.c:133-152`).
//! - [`idct4x4_16_add`] — 2D 16-coefficient inverse 4×4 DCT with
//!   residual add. Mirrors `vpx_idct4x4_16_add_c`
//!   (`vpx_dsp/inv_txfm.c:154-176`).
//! - [`idct4x4_1_add`] — DC-only fast path. Mirrors
//!   `vpx_idct4x4_1_add_c` (`vpx_dsp/inv_txfm.c:178-195`).
//!
//! 8×8 / 16×16 / 32×32 inverse DCT, plus the iadst 4/8/16 family and
//! the hybrid 4×4/8×8/16×16 transforms, land in follow-up sub-tasks.

use super::constants::{COSPI_8_64, COSPI_16_64, COSPI_24_64, clip_pixel_add, dct_const_round_shift};

/// 1D 4-point inverse DCT. Output values are in the
/// post-`WRAPLOW`-and-shift domain (effectively 16-bit signed, but we
/// hold them in `i32` to avoid spurious truncation between the row and
/// column passes).
///
/// Mirrors libvpx `idct4_c` (`vpx_dsp/inv_txfm.c:133-152`) for the
/// non-HBD path (`WRAPLOW` is a no-op outside `CONFIG_EMULATE_HARDWARE`
/// in libvpx default builds — `inv_txfm.h:87`).
#[inline]
pub fn idct4(input: &[i32; 4], output: &mut [i32; 4]) {
    // stage 1
    // temp1 = (input[0] + input[2]) * COSPI_16_64
    let temp1: i64 = (input[0] as i64 + input[2] as i64) * COSPI_16_64 as i64;
    let temp2: i64 = (input[0] as i64 - input[2] as i64) * COSPI_16_64 as i64;
    let step0: i32 = dct_const_round_shift(temp1) as i32;
    let step1: i32 = dct_const_round_shift(temp2) as i32;
    let temp1: i64 =
        input[1] as i64 * COSPI_24_64 as i64 - input[3] as i64 * COSPI_8_64 as i64;
    let temp2: i64 =
        input[1] as i64 * COSPI_8_64 as i64 + input[3] as i64 * COSPI_24_64 as i64;
    let step2: i32 = dct_const_round_shift(temp1) as i32;
    let step3: i32 = dct_const_round_shift(temp2) as i32;

    // stage 2
    output[0] = step0 + step3;
    output[1] = step1 + step2;
    output[2] = step1 - step2;
    output[3] = step0 - step3;
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

    // Two passes of (input * COSPI_16_64) and a >>4 final shift:
    // out = dct_const_round_shift(dct_const_round_shift(input * cospi_16_64) * cospi_16_64)
    // a1  = (out + 8) >> 4
    let out1 = dct_const_round_shift((input as i64) * (COSPI_16_64 as i64));
    let out2 = dct_const_round_shift(out1 * (COSPI_16_64 as i64));
    let a1 = (out2 + 8) >> 4;
    for r in 0..4 {
        for c in 0..4 {
            dest[r * stride + c] = clip_pixel_add(dest[r * stride + c], a1);
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
}
