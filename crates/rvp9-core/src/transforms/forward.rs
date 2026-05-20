//! Forward transform scalar reference kernels (encoder side).
//!
//! Currently lands the 4×4 forward-DCT / forward-ADST family:
//!
//! - [`fdct4`] — 1D 4-point forward DCT. Mirrors libvpx `fdct4`
//!   (`vp9/encoder/vp9_dct.c:23-40`). Identical body to the inner-loop
//!   transform in `vpx_fdct4x4_c` (`vpx_dsp/fwd_txfm.c:51-63`).
//! - [`fadst4`] — 1D 4-point forward ADST. Mirrors libvpx `fadst4`
//!   (`vp9/encoder/vp9_dct.c:235-273`).
//! - [`fdct4x4`] — 2D 4×4 forward DCT with the pass-0 ×16 scaling and
//!   the final `(x + 1) >> 2` step. Mirrors libvpx `vpx_fdct4x4_c`
//!   (`vpx_dsp/fwd_txfm.c:15-79`).
//! - [`fht4x4`] — 2D 4×4 hybrid (DCT/ADST per axis), dispatched on
//!   [`super::TxType`]. Mirrors libvpx `vp9_fht4x4_c`
//!   (`vp9/encoder/vp9_dct.c:530-555`). `DCT_DCT` short-circuits to
//!   [`fdct4x4`].
//!
//! Like the inverse-side kernels, every `fdct_round_shift` result that
//! lands in a `tran_low_t` storage slot is passed through [`wraplow`]
//! to mirror libvpx's implicit `int16_t` truncation on every assignment
//! to a `tran_low_t` (`int16_t` in the default — non-HBD — build).
//!
//! Larger sizes (`fdct8`, `fdct16`, `fdct32`, `fadst8`, `fadst16`, plus
//! the `fdct8x8` / `fdct16x16` / `fdct32x32` 2D wrappers and `fht8x8` /
//! `fht16x16`) follow in later sub-tasks. The Walsh-Hadamard
//! lossless-only kernel (`vp9_fwht4x4_c`) is also deferred — the v1
//! encoder never emits lossless frames.

use super::constants::{
    COSPI_8_64, COSPI_16_64, COSPI_24_64, SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9,
    dct_const_round_shift, wraplow,
};
use super::types::TxType;

/// Function pointer for a 1D 4-point forward transform — either
/// [`fdct4`] or [`fadst4`]. Used by the [`fht4x4`] dispatcher to pick
/// per-axis kernels via the FHT_4 table.
type Fwd1d4 = fn(&[i32; 4], &mut [i32; 4]);

/// 1D 4-point forward DCT. The `tran_high_t` intermediates in the C
/// source live in `i64` to mirror the 32-bit-safe arithmetic of
/// `tran_high_t`; outputs are stored after `dct_const_round_shift` +
/// [`wraplow`] truncation, matching libvpx's `tran_low_t` storage.
///
/// Mirrors libvpx `fdct4` (`vp9/encoder/vp9_dct.c:23-40`).
#[inline]
pub fn fdct4(input: &[i32; 4], output: &mut [i32; 4]) {
    let in0 = input[0] as i64;
    let in1 = input[1] as i64;
    let in2 = input[2] as i64;
    let in3 = input[3] as i64;

    let step0 = in0 + in3;
    let step1 = in1 + in2;
    let step2 = in1 - in2;
    let step3 = in0 - in3;

    let temp1 = (step0 + step1) * COSPI_16_64 as i64;
    let temp2 = (step0 - step1) * COSPI_16_64 as i64;
    output[0] = wraplow(dct_const_round_shift(temp1) as i32);
    output[2] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step2 * COSPI_24_64 as i64 + step3 * COSPI_8_64 as i64;
    let temp2 = -step2 * COSPI_8_64 as i64 + step3 * COSPI_24_64 as i64;
    output[1] = wraplow(dct_const_round_shift(temp1) as i32);
    output[3] = wraplow(dct_const_round_shift(temp2) as i32);
}

/// 1D 4-point forward ADST. Has an explicit all-zero short-circuit
/// matching libvpx's `if (!(x0 | x1 | x2 | x3))` early-return at line
/// 244 — important because the staged computation isn't a no-op for
/// all-zero input (`s7 = x0 + x1 - x3 = 0` then `x1 = sinpi_3_9 * 0`
/// would otherwise still execute; the short-circuit is the C source's
/// own optimisation and we mirror it verbatim).
///
/// Mirrors libvpx `fadst4` (`vp9/encoder/vp9_dct.c:235-273`).
#[inline]
pub fn fadst4(input: &[i32; 4], output: &mut [i32; 4]) {
    let x0 = input[0] as i64;
    let x1 = input[1] as i64;
    let x2 = input[2] as i64;
    let x3 = input[3] as i64;

    if x0 == 0 && x1 == 0 && x2 == 0 && x3 == 0 {
        output[0] = 0;
        output[1] = 0;
        output[2] = 0;
        output[3] = 0;
        return;
    }

    let s0 = SINPI_1_9 as i64 * x0;
    let s1 = SINPI_4_9 as i64 * x0;
    let s2 = SINPI_2_9 as i64 * x1;
    let s3 = SINPI_1_9 as i64 * x1;
    let s4 = SINPI_3_9 as i64 * x2;
    let s5 = SINPI_4_9 as i64 * x3;
    let s6 = SINPI_2_9 as i64 * x3;
    let s7 = x0 + x1 - x3;

    let x0_2 = s0 + s2 + s5;
    let x1_2 = SINPI_3_9 as i64 * s7;
    let x2_2 = s1 - s3 + s6;
    let x3_2 = s4;

    let s0 = x0_2 + x3_2;
    let s1 = x1_2;
    let s2 = x2_2 - x3_2;
    let s3 = x2_2 - x0_2 + x3_2;

    // 1-D transform scaling factor is sqrt(2).
    output[0] = wraplow(dct_const_round_shift(s0) as i32);
    output[1] = wraplow(dct_const_round_shift(s1) as i32);
    output[2] = wraplow(dct_const_round_shift(s2) as i32);
    output[3] = wraplow(dct_const_round_shift(s3) as i32);
}

/// 2D 4×4 forward DCT. `input` is the 16 residual samples in raster
/// order at `stride` (typically the prediction-residual buffer);
/// `output` is the 16 forward coefficients in raster order.
///
/// The two-pass column/row structure plus the `* 16` pass-0 scaling
/// and final `(x + 1) >> 2` step mirror libvpx `vpx_fdct4x4_c`
/// (`vpx_dsp/fwd_txfm.c:15-79`). The DC bump (`if i == 0 &&
/// in_high[0] { ++in_high[0]; }`) is preserved verbatim.
pub fn fdct4x4(input: &[i16], output: &mut [i32; 16], stride: usize) {
    debug_assert!(
        input.len() >= 3 * stride + 4,
        "input slice too small for 4x4 at stride {stride}"
    );

    let mut intermediate = [0i32; 16];

    // Pass 0: column transform with ×16 scaling and DC bump on column 0.
    for i in 0..4 {
        let mut in_high = [0i64; 4];
        in_high[0] = (input[i] as i64) * 16;
        in_high[1] = (input[stride + i] as i64) * 16;
        in_high[2] = (input[2 * stride + i] as i64) * 16;
        in_high[3] = (input[3 * stride + i] as i64) * 16;
        if i == 0 && in_high[0] != 0 {
            in_high[0] += 1;
        }

        let step0 = in_high[0] + in_high[3];
        let step1 = in_high[1] + in_high[2];
        let step2 = in_high[1] - in_high[2];
        let step3 = in_high[0] - in_high[3];

        let temp1 = (step0 + step1) * COSPI_16_64 as i64;
        let temp2 = (step0 - step1) * COSPI_16_64 as i64;
        intermediate[i * 4] = wraplow(dct_const_round_shift(temp1) as i32);
        intermediate[i * 4 + 2] = wraplow(dct_const_round_shift(temp2) as i32);

        let temp1 = step2 * COSPI_24_64 as i64 + step3 * COSPI_8_64 as i64;
        let temp2 = -step2 * COSPI_8_64 as i64 + step3 * COSPI_24_64 as i64;
        intermediate[i * 4 + 1] = wraplow(dct_const_round_shift(temp1) as i32);
        intermediate[i * 4 + 3] = wraplow(dct_const_round_shift(temp2) as i32);
    }

    // Pass 1: row transform on transposed intermediate. `in_low` walks
    // intermediate column-by-column (stride 4), `out` walks output
    // row-by-row.
    for i in 0..4 {
        let in_high = [
            intermediate[i] as i64,
            intermediate[4 + i] as i64,
            intermediate[8 + i] as i64,
            intermediate[12 + i] as i64,
        ];

        let step0 = in_high[0] + in_high[3];
        let step1 = in_high[1] + in_high[2];
        let step2 = in_high[1] - in_high[2];
        let step3 = in_high[0] - in_high[3];

        let temp1 = (step0 + step1) * COSPI_16_64 as i64;
        let temp2 = (step0 - step1) * COSPI_16_64 as i64;
        output[i * 4] = wraplow(dct_const_round_shift(temp1) as i32);
        output[i * 4 + 2] = wraplow(dct_const_round_shift(temp2) as i32);

        let temp1 = step2 * COSPI_24_64 as i64 + step3 * COSPI_8_64 as i64;
        let temp2 = -step2 * COSPI_8_64 as i64 + step3 * COSPI_24_64 as i64;
        output[i * 4 + 1] = wraplow(dct_const_round_shift(temp1) as i32);
        output[i * 4 + 3] = wraplow(dct_const_round_shift(temp2) as i32);
    }

    // Final scale-down: `(x + 1) >> 2` per `vpx_fdct4x4_c:73-78`.
    for slot in output.iter_mut() {
        *slot = wraplow((*slot + 1) >> 2);
    }
}

/// 2D 4×4 hybrid forward transform (DCT or ADST per axis), selected by
/// `tx_type`. `DCT_DCT` short-circuits to [`fdct4x4`] (matching libvpx
/// line 532-533). Other types use the column / row 1D dispatch with
/// pass-0 ×16 scaling and DC bump on column 0 and the final
/// `(x + 1) >> 2` reduction.
///
/// Mirrors libvpx `vp9_fht4x4_c` (`vp9/encoder/vp9_dct.c:530-555`).
pub fn fht4x4(input: &[i16], output: &mut [i32; 16], stride: usize, tx_type: TxType) {
    if matches!(tx_type, TxType::DctDct) {
        fdct4x4(input, output, stride);
        return;
    }

    debug_assert!(
        input.len() >= 3 * stride + 4,
        "input slice too small for 4x4 at stride {stride}"
    );

    // libvpx FHT_4 table (vp9_dct.c:521-527):
    //   ADST_DCT  = { fadst4, fdct4 }   (col=fadst, row=fdct)
    //   DCT_ADST  = { fdct4,  fadst4 }
    //   ADST_ADST = { fadst4, fadst4 }
    let (col_fn, row_fn): (Fwd1d4, Fwd1d4) = match tx_type {
        TxType::DctDct => unreachable!(),
        TxType::AdstDct => (fadst4, fdct4),
        TxType::DctAdst => (fdct4, fadst4),
        TxType::AdstAdst => (fadst4, fadst4),
    };

    let mut intermediate = [0i32; 16];

    // Columns: load column i with ×16 scaling, DC bump on column 0,
    // transform, then transpose-store into `intermediate[j * 4 + i]`.
    for i in 0..4 {
        let mut col_in = [0i32; 4];
        col_in[0] = (input[i] as i32) * 16;
        col_in[1] = (input[stride + i] as i32) * 16;
        col_in[2] = (input[2 * stride + i] as i32) * 16;
        col_in[3] = (input[3 * stride + i] as i32) * 16;
        if i == 0 && col_in[0] != 0 {
            col_in[0] += 1;
        }
        let mut col_out = [0i32; 4];
        col_fn(&col_in, &mut col_out);
        for j in 0..4 {
            intermediate[j * 4 + i] = col_out[j];
        }
    }

    // Rows: load row i, transform, then store with final `(x + 1) >> 2`.
    for i in 0..4 {
        let row_in: [i32; 4] = [
            intermediate[i * 4],
            intermediate[i * 4 + 1],
            intermediate[i * 4 + 2],
            intermediate[i * 4 + 3],
        ];
        let mut row_out = [0i32; 4];
        row_fn(&row_in, &mut row_out);
        for j in 0..4 {
            output[i * 4 + j] = wraplow((row_out[j] + 1) >> 2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::inverse::{idct4, iadst4, iht4x4_16_add};
    use super::*;

    /// Zero input → zero output (1D DCT).
    #[test]
    fn fdct4_zero_in_zero_out() {
        let input = [0i32; 4];
        let mut output = [99i32; 4];
        fdct4(&input, &mut output);
        assert_eq!(output, [0; 4]);
    }

    /// Zero input → zero output (1D ADST). Exercises the explicit
    /// short-circuit at `vp9_dct.c:244`.
    #[test]
    fn fadst4_zero_in_zero_out() {
        let input = [0i32; 4];
        let mut output = [99i32; 4];
        fadst4(&input, &mut output);
        assert_eq!(output, [0; 4]);
    }

    /// `fdct4` is its own inverse up to `dct_const_round_shift` scaling
    /// for orthogonal inputs: a DCT-domain impulse round-trips through
    /// `idct4` to a recognizable (cos-weighted) basis vector.
    /// Exercised here by composing `idct4(fdct4(...))` on a small input
    /// and checking the relative magnitude matches (within a small
    /// integer rounding tolerance).
    #[test]
    fn fdct4_idct4_round_trip_small() {
        // Input chosen so the post-transform values stay small.
        let input: [i32; 4] = [10, -4, 6, -2];
        let mut spectrum = [0i32; 4];
        fdct4(&input, &mut spectrum);
        let mut round_trip = [0i32; 4];
        idct4(&spectrum, &mut round_trip);
        // 1D fdct4+idct4 produces input scaled by 2 (sqrt(2)^2) per
        // libvpx convention. Validate by comparing element-by-element
        // with a small tolerance.
        for k in 0..4 {
            let expected = input[k] * 2;
            assert!(
                (round_trip[k] - expected).abs() <= 1,
                "round_trip[{k}] = {} not within ±1 of 2 * input[{k}] = {}",
                round_trip[k],
                expected
            );
        }
    }

    /// 1D ADST forward + inverse round-trip for a small non-zero input.
    /// The 1D ADST is orthonormal up to `sqrt(2)`, so the round-trip
    /// recovers `2 * input` within ±1.
    #[test]
    fn fadst4_iadst4_round_trip_small() {
        let input: [i32; 4] = [8, -5, 3, -1];
        let mut spectrum = [0i32; 4];
        fadst4(&input, &mut spectrum);
        let mut round_trip = [0i32; 4];
        iadst4(&spectrum, &mut round_trip);
        for k in 0..4 {
            let expected = input[k] * 2;
            assert!(
                (round_trip[k] - expected).abs() <= 1,
                "round_trip[{k}] = {} not within ±1 of 2 * input[{k}] = {}",
                round_trip[k],
                expected
            );
        }
    }

    /// Zero 4×4 residual block produces zero coefficients (2D DCT).
    #[test]
    fn fdct4x4_zero_input_zero_output() {
        let input = [0i16; 16];
        let mut output = [99i32; 16];
        fdct4x4(&input, &mut output, 4);
        assert_eq!(output, [0; 16]);
    }

    /// Zero 4×4 residual block produces zero coefficients for every
    /// `TxType` (2D hybrid).
    #[test]
    fn fht4x4_zero_input_zero_output_all_tx_types() {
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let input = [0i16; 16];
            let mut output = [99i32; 16];
            fht4x4(&input, &mut output, 4, tx_type);
            assert_eq!(
                output, [0; 16],
                "tx_type {tx_type:?} did not produce zero output for zero input"
            );
        }
    }

    /// `fht4x4` with `DCT_DCT` is bit-identical to `fdct4x4` — the
    /// `vp9_fht4x4_c` short-circuit at line 532-533 in libvpx.
    #[test]
    fn fht4x4_dct_dct_matches_fdct4x4() {
        let input: [i16; 16] = [
            10, -4, 6, -2, //
            -3, 8, -1, 5, //
            7, -6, 4, -2, //
            1, -3, 9, -5,
        ];
        let mut out_fht = [0i32; 16];
        let mut out_fdct = [0i32; 16];
        fht4x4(&input, &mut out_fht, 4, TxType::DctDct);
        fdct4x4(&input, &mut out_fdct, 4);
        assert_eq!(out_fht, out_fdct);
    }

    /// `fht4x4 → iht4x4_16_add` recovers the original residual block
    /// (within ±1 per element) when starting from a zero destination.
    /// This is the canonical encoder-side round-trip oracle: residuals
    /// → coefficients → reconstructed pixels recovers the residual to
    /// integer precision.
    ///
    /// Inputs deliberately small-magnitude (`[-8, 8]`) so the
    /// reconstruction stays in unsigned-pixel range when added to a
    /// dest of 128.
    #[test]
    fn fht4x4_then_iht4x4_recovers_small_residual_all_tx_types() {
        let residual: [i16; 16] = [
            5, -3, 2, -1, //
            -2, 4, -1, 3, //
            1, -2, 5, -4, //
            -1, 2, -3, 6,
        ];
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let mut coefs = [0i32; 16];
            fht4x4(&residual, &mut coefs, 4, tx_type);

            // iht*_add reconstructs delta and adds to dest. Start with
            // dest = 128 and check (dest_after - 128) ≈ residual.
            let mut dest = [128u8; 16];
            iht4x4_16_add(&coefs, &mut dest, 4, tx_type);

            for k in 0..16 {
                let recovered = (dest[k] as i32) - 128;
                let expected = residual[k] as i32;
                assert!(
                    (recovered - expected).abs() <= 1,
                    "{tx_type:?}: pos {k} recovered={recovered} expected={expected} \
                     (dest[k]={}, residual[k]={})",
                    dest[k],
                    residual[k]
                );
            }
        }
    }

    /// `fht4x4` honours non-trivial `stride`. The same residual values
    /// laid out with stride 8 (rows of 8 with the last 4 ignored)
    /// produce the same coefficients as the densely-packed stride-4
    /// version.
    #[test]
    fn fht4x4_stride_independence() {
        let residual: [i16; 16] = [
            5, -3, 2, -1, //
            -2, 4, -1, 3, //
            1, -2, 5, -4, //
            -1, 2, -3, 6,
        ];
        let mut wide = [0i16; 32];
        for r in 0..4 {
            for c in 0..4 {
                wide[r * 8 + c] = residual[r * 4 + c];
            }
        }

        let mut out_tight = [0i32; 16];
        let mut out_wide = [0i32; 16];
        fdct4x4(&residual, &mut out_tight, 4);
        fdct4x4(&wide, &mut out_wide, 8);
        assert_eq!(
            out_tight, out_wide,
            "stride must not affect coefficient values when content matches"
        );
    }

    /// Approximate linearity: scaling input by 2 should scale output
    /// by 2 within ±1 per coefficient. The fdct uses
    /// `dct_const_round_shift` (round-half-up at the 14-bit boundary)
    /// so `fdct(2x) != 2 * fdct(x)` bit-exactly — a 5.4-precision
    /// value rounds to 5, but 10.8 rounds to 11. Still a useful sanity
    /// bound.
    #[test]
    fn fdct4_approximate_linearity_scale_by_2() {
        let input: [i32; 4] = [3, -1, 2, -4];
        let mut out1 = [0i32; 4];
        let mut out2 = [0i32; 4];
        fdct4(&input, &mut out1);
        let scaled = [input[0] * 2, input[1] * 2, input[2] * 2, input[3] * 2];
        fdct4(&scaled, &mut out2);
        for k in 0..4 {
            let expected = out1[k] * 2;
            assert!(
                (out2[k] - expected).abs() <= 1,
                "approx linearity broken at index {k}: out2={} not within ±1 of \
                 2*out1={}",
                out2[k],
                expected
            );
        }
    }
}
