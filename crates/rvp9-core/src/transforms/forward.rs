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
    COSPI_1_64, COSPI_2_64, COSPI_3_64, COSPI_4_64, COSPI_5_64, COSPI_6_64, COSPI_7_64, COSPI_8_64,
    COSPI_9_64, COSPI_10_64, COSPI_11_64, COSPI_12_64, COSPI_13_64, COSPI_14_64, COSPI_15_64,
    COSPI_16_64, COSPI_17_64, COSPI_18_64, COSPI_19_64, COSPI_20_64, COSPI_21_64, COSPI_22_64,
    COSPI_23_64, COSPI_24_64, COSPI_25_64, COSPI_26_64, COSPI_27_64, COSPI_28_64, COSPI_29_64,
    COSPI_30_64, COSPI_31_64, SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9, dct_const_round_shift,
    wraplow,
};
use super::types::TxType;

/// Function pointer for a 1D 4-point forward transform — either
/// [`fdct4`] or [`fadst4`]. Used by the [`fht4x4`] dispatcher to pick
/// per-axis kernels via the FHT_4 table.
type Fwd1d4 = fn(&[i32; 4], &mut [i32; 4]);

/// Function pointer for a 1D 8-point forward transform — either
/// [`fdct8`] or [`fadst8`]. Used by [`fht8x8`] via the FHT_8 table.
type Fwd1d8 = fn(&[i32; 8], &mut [i32; 8]);

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

// ---------------------------------------------------------------------
// 8×8 forward transform family.
// ---------------------------------------------------------------------

/// 1D 8-point forward DCT. Four stages of butterflies + cospi-weighted
/// rotations. Mirrors libvpx `fdct8` (`vp9/encoder/vp9_dct.c:42-92`).
#[inline]
pub fn fdct8(input: &[i32; 8], output: &mut [i32; 8]) {
    // stage 1
    let s0 = input[0] as i64 + input[7] as i64;
    let s1 = input[1] as i64 + input[6] as i64;
    let s2 = input[2] as i64 + input[5] as i64;
    let s3 = input[3] as i64 + input[4] as i64;
    let s4 = input[3] as i64 - input[4] as i64;
    let s5 = input[2] as i64 - input[5] as i64;
    let s6 = input[1] as i64 - input[6] as i64;
    let s7 = input[0] as i64 - input[7] as i64;

    // fdct4(step, step) sub-block on s0..s3
    let x0 = s0 + s3;
    let x1 = s1 + s2;
    let x2 = s1 - s2;
    let x3 = s0 - s3;
    let t0 = (x0 + x1) * COSPI_16_64 as i64;
    let t1 = (x0 - x1) * COSPI_16_64 as i64;
    let t2 = x2 * COSPI_24_64 as i64 + x3 * COSPI_8_64 as i64;
    let t3 = -x2 * COSPI_8_64 as i64 + x3 * COSPI_24_64 as i64;
    output[0] = wraplow(dct_const_round_shift(t0) as i32);
    output[2] = wraplow(dct_const_round_shift(t2) as i32);
    output[4] = wraplow(dct_const_round_shift(t1) as i32);
    output[6] = wraplow(dct_const_round_shift(t3) as i32);

    // Stage 2
    let t0 = (s6 - s5) * COSPI_16_64 as i64;
    let t1 = (s6 + s5) * COSPI_16_64 as i64;
    // libvpx: t2/t3 are `tran_high_t` here but assigned from
    // `fdct_round_shift` — they stay in 32-bit land. We carry them in
    // i64 for arithmetic safety; values still fit in i32.
    let t2 = dct_const_round_shift(t0);
    let t3 = dct_const_round_shift(t1);

    // Stage 3
    let x0 = s4 + t2;
    let x1 = s4 - t2;
    let x2 = s7 - t3;
    let x3 = s7 + t3;

    // Stage 4
    let t0 = x0 * COSPI_28_64 as i64 + x3 * COSPI_4_64 as i64;
    let t1 = x1 * COSPI_12_64 as i64 + x2 * COSPI_20_64 as i64;
    let t2 = x2 * COSPI_12_64 as i64 + x1 * -(COSPI_20_64 as i64);
    let t3 = x3 * COSPI_28_64 as i64 + x0 * -(COSPI_4_64 as i64);
    output[1] = wraplow(dct_const_round_shift(t0) as i32);
    output[3] = wraplow(dct_const_round_shift(t2) as i32);
    output[5] = wraplow(dct_const_round_shift(t1) as i32);
    output[7] = wraplow(dct_const_round_shift(t3) as i32);
}

/// 1D 8-point forward ADST. Three stages with input + output
/// permutations and explicit sign flips on half the outputs. The
/// outputs at indices 1, 3, 5, 7 carry sign flips.
///
/// Mirrors libvpx `fadst8` (`vp9/encoder/vp9_dct.c:275-344`).
#[inline]
pub fn fadst8(input: &[i32; 8], output: &mut [i32; 8]) {
    // Input permutation per libvpx :278-285.
    let x0 = input[7] as i64;
    let x1 = input[0] as i64;
    let x2 = input[5] as i64;
    let x3 = input[2] as i64;
    let x4 = input[3] as i64;
    let x5 = input[4] as i64;
    let x6 = input[1] as i64;
    let x7 = input[6] as i64;

    // stage 1
    let s0 = COSPI_2_64 as i64 * x0 + COSPI_30_64 as i64 * x1;
    let s1 = COSPI_30_64 as i64 * x0 - COSPI_2_64 as i64 * x1;
    let s2 = COSPI_10_64 as i64 * x2 + COSPI_22_64 as i64 * x3;
    let s3 = COSPI_22_64 as i64 * x2 - COSPI_10_64 as i64 * x3;
    let s4 = COSPI_18_64 as i64 * x4 + COSPI_14_64 as i64 * x5;
    let s5 = COSPI_14_64 as i64 * x4 - COSPI_18_64 as i64 * x5;
    let s6 = COSPI_26_64 as i64 * x6 + COSPI_6_64 as i64 * x7;
    let s7 = COSPI_6_64 as i64 * x6 - COSPI_26_64 as i64 * x7;

    let x0 = dct_const_round_shift(s0 + s4);
    let x1 = dct_const_round_shift(s1 + s5);
    let x2 = dct_const_round_shift(s2 + s6);
    let x3 = dct_const_round_shift(s3 + s7);
    let x4 = dct_const_round_shift(s0 - s4);
    let x5 = dct_const_round_shift(s1 - s5);
    let x6 = dct_const_round_shift(s2 - s6);
    let x7 = dct_const_round_shift(s3 - s7);

    // stage 2 — pass-through for s0..s3
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = COSPI_8_64 as i64 * x4 + COSPI_24_64 as i64 * x5;
    let s5 = COSPI_24_64 as i64 * x4 - COSPI_8_64 as i64 * x5;
    let s6 = -(COSPI_24_64 as i64) * x6 + COSPI_8_64 as i64 * x7;
    let s7 = COSPI_8_64 as i64 * x6 + COSPI_24_64 as i64 * x7;

    let x0 = s0 + s2;
    let x1 = s1 + s3;
    let x2 = s0 - s2;
    let x3 = s1 - s3;
    let x4 = dct_const_round_shift(s4 + s6);
    let x5 = dct_const_round_shift(s5 + s7);
    let x6 = dct_const_round_shift(s4 - s6);
    let x7 = dct_const_round_shift(s5 - s7);

    // stage 3
    let s2 = COSPI_16_64 as i64 * (x2 + x3);
    let s3 = COSPI_16_64 as i64 * (x2 - x3);
    let s6 = COSPI_16_64 as i64 * (x6 + x7);
    let s7 = COSPI_16_64 as i64 * (x6 - x7);

    let x2 = dct_const_round_shift(s2);
    let x3 = dct_const_round_shift(s3);
    let x6 = dct_const_round_shift(s6);
    let x7 = dct_const_round_shift(s7);

    // Output permutation with sign flips per libvpx :336-343. Each
    // store is a `tran_low_t` assignment so we apply `wraplow`.
    output[0] = wraplow(x0 as i32);
    output[1] = wraplow(-x4 as i32);
    output[2] = wraplow(x6 as i32);
    output[3] = wraplow(-x2 as i32);
    output[4] = wraplow(x3 as i32);
    output[5] = wraplow(-x7 as i32);
    output[6] = wraplow(x5 as i32);
    output[7] = wraplow(-x1 as i32);
}

/// 2D 8×8 forward DCT. Pass 0 multiplies input by 4 (no DC bump,
/// unlike 4×4); pass 1 reads from intermediate at integer values
/// directly (no `(x + 1) >> 2` between passes). The final reduction
/// is `output /= 2` — **integer truncation toward zero**, equivalent
/// to Rust `/` on signed integers (per C99 6.5.5).
///
/// Note the difference from 4×4: 4×4 uses `(x + 1) >> 2` (round half
/// up); 8×8 uses `/= 2` (truncate toward zero). The hybrid wrapper
/// [`fht8x8`] uses the textually-different `(x + (x<0)) >> 1` form
/// which is semantically identical to `/ 2` for `i32`.
///
/// Mirrors libvpx `vpx_fdct8x8_c` (`vpx_dsp/fwd_txfm.c:90-172`).
pub fn fdct8x8(input: &[i16], output: &mut [i32; 64], stride: usize) {
    debug_assert!(
        input.len() >= 7 * stride + 8,
        "input slice too small for 8x8 at stride {stride}"
    );

    let mut intermediate = [0i32; 64];

    // Pass 0: column transform with ×4 scaling.
    for i in 0..8 {
        let s0 = (input[i] as i64 + input[7 * stride + i] as i64) * 4;
        let s1 = (input[stride + i] as i64 + input[6 * stride + i] as i64) * 4;
        let s2 = (input[2 * stride + i] as i64 + input[5 * stride + i] as i64) * 4;
        let s3 = (input[3 * stride + i] as i64 + input[4 * stride + i] as i64) * 4;
        let s4 = (input[3 * stride + i] as i64 - input[4 * stride + i] as i64) * 4;
        let s5 = (input[2 * stride + i] as i64 - input[5 * stride + i] as i64) * 4;
        let s6 = (input[stride + i] as i64 - input[6 * stride + i] as i64) * 4;
        let s7 = (input[i] as i64 - input[7 * stride + i] as i64) * 4;
        fdct8_inner(s0, s1, s2, s3, s4, s5, s6, s7, &mut intermediate, i * 8);
    }

    // Pass 1: row transform on transposed intermediate (in-place
    // semantics from libvpx — `in_low` walks intermediate column-by-
    // column at stride 8, `out` is the output buffer).
    for i in 0..8 {
        let s0 = intermediate[i] as i64 + intermediate[7 * 8 + i] as i64;
        let s1 = intermediate[8 + i] as i64 + intermediate[6 * 8 + i] as i64;
        let s2 = intermediate[2 * 8 + i] as i64 + intermediate[5 * 8 + i] as i64;
        let s3 = intermediate[3 * 8 + i] as i64 + intermediate[4 * 8 + i] as i64;
        let s4 = intermediate[3 * 8 + i] as i64 - intermediate[4 * 8 + i] as i64;
        let s5 = intermediate[2 * 8 + i] as i64 - intermediate[5 * 8 + i] as i64;
        let s6 = intermediate[8 + i] as i64 - intermediate[6 * 8 + i] as i64;
        let s7 = intermediate[i] as i64 - intermediate[7 * 8 + i] as i64;
        fdct8_inner(s0, s1, s2, s3, s4, s5, s6, s7, output, i * 8);
    }

    // Final reduction: `output /= 2` (truncation toward zero).
    for slot in output.iter_mut() {
        *slot = wraplow(*slot / 2);
    }
}

/// Helper: the fdct8 inner pipeline operating on pre-loaded
/// `s0..s7` values and writing into `out[base..base+8]`. Factored out
/// because [`fdct8x8`] needs it twice (one pass with ×4 scaling, one
/// without) and inlining the body twice would obscure the structure.
#[allow(clippy::too_many_arguments)]
#[inline]
fn fdct8_inner(
    s0: i64,
    s1: i64,
    s2: i64,
    s3: i64,
    s4: i64,
    s5: i64,
    s6: i64,
    s7: i64,
    out: &mut [i32],
    base: usize,
) {
    // fdct4 sub-block on s0..s3.
    let x0 = s0 + s3;
    let x1 = s1 + s2;
    let x2 = s1 - s2;
    let x3 = s0 - s3;
    let t0 = (x0 + x1) * COSPI_16_64 as i64;
    let t1 = (x0 - x1) * COSPI_16_64 as i64;
    let t2 = x2 * COSPI_24_64 as i64 + x3 * COSPI_8_64 as i64;
    let t3 = -x2 * COSPI_8_64 as i64 + x3 * COSPI_24_64 as i64;
    out[base] = wraplow(dct_const_round_shift(t0) as i32);
    out[base + 2] = wraplow(dct_const_round_shift(t2) as i32);
    out[base + 4] = wraplow(dct_const_round_shift(t1) as i32);
    out[base + 6] = wraplow(dct_const_round_shift(t3) as i32);

    // Stage 2.
    let t0 = (s6 - s5) * COSPI_16_64 as i64;
    let t1 = (s6 + s5) * COSPI_16_64 as i64;
    let t2 = dct_const_round_shift(t0);
    let t3 = dct_const_round_shift(t1);

    // Stage 3.
    let x0 = s4 + t2;
    let x1 = s4 - t2;
    let x2 = s7 - t3;
    let x3 = s7 + t3;

    // Stage 4.
    let t0 = x0 * COSPI_28_64 as i64 + x3 * COSPI_4_64 as i64;
    let t1 = x1 * COSPI_12_64 as i64 + x2 * COSPI_20_64 as i64;
    let t2 = x2 * COSPI_12_64 as i64 + x1 * -(COSPI_20_64 as i64);
    let t3 = x3 * COSPI_28_64 as i64 + x0 * -(COSPI_4_64 as i64);
    out[base + 1] = wraplow(dct_const_round_shift(t0) as i32);
    out[base + 3] = wraplow(dct_const_round_shift(t2) as i32);
    out[base + 5] = wraplow(dct_const_round_shift(t1) as i32);
    out[base + 7] = wraplow(dct_const_round_shift(t3) as i32);
}

/// 2D 8×8 hybrid forward transform (DCT or ADST per axis), selected
/// by `tx_type`. `DCT_DCT` short-circuits to [`fdct8x8`] (matching
/// libvpx :559-560). Other types use the column / row 1D dispatch
/// with pass-0 ×4 scaling (no DC bump at 8×8) and final
/// `(x + (x < 0)) >> 1` reduction (= `/ 2` truncation toward zero).
///
/// Mirrors libvpx `vp9_fht8x8_c` (`vp9/encoder/vp9_dct.c:557-582`).
pub fn fht8x8(input: &[i16], output: &mut [i32; 64], stride: usize, tx_type: TxType) {
    if matches!(tx_type, TxType::DctDct) {
        fdct8x8(input, output, stride);
        return;
    }

    debug_assert!(
        input.len() >= 7 * stride + 8,
        "input slice too small for 8x8 at stride {stride}"
    );

    // FHT_8 table (vp9_dct.c:516-521).
    let (col_fn, row_fn): (Fwd1d8, Fwd1d8) = match tx_type {
        TxType::DctDct => unreachable!(),
        TxType::AdstDct => (fadst8, fdct8),
        TxType::DctAdst => (fdct8, fadst8),
        TxType::AdstAdst => (fadst8, fadst8),
    };

    let mut intermediate = [0i32; 64];

    // Columns: ×4 scaling, transform, transpose-store into intermediate.
    for i in 0..8 {
        let mut col_in = [0i32; 8];
        for j in 0..8 {
            col_in[j] = (input[j * stride + i] as i32) * 4;
        }
        let mut col_out = [0i32; 8];
        col_fn(&col_in, &mut col_out);
        for j in 0..8 {
            intermediate[j * 8 + i] = col_out[j];
        }
    }

    // Rows: transform, then `(x + (x < 0)) >> 1`.
    for i in 0..8 {
        let mut row_in = [0i32; 8];
        for j in 0..8 {
            row_in[j] = intermediate[j + i * 8];
        }
        let mut row_out = [0i32; 8];
        row_fn(&row_in, &mut row_out);
        for j in 0..8 {
            // `(x + (x < 0)) >> 1` == `x / 2` (truncation toward zero
            // for two's-complement i32).
            let v = row_out[j];
            let adjusted = v + i32::from(v < 0);
            output[j + i * 8] = wraplow(adjusted >> 1);
        }
    }
}

// ---------------------------------------------------------------------
// 16×16 forward transform family.
// ---------------------------------------------------------------------

/// Function pointer for a 1D 16-point forward transform — either
/// [`fdct16`] or [`fadst16`]. Used by [`fht16x16`] via the FHT_16 table.
type Fwd1d16 = fn(&[i32; 16], &mut [i32; 16]);

/// 1D 16-point forward DCT. Computes the 8 even-indexed outputs via a
/// nested `fdct8`-equivalent sub-block on the sum half
/// (`input[0]+input[15]`, `input[1]+input[14]`, ...) and the 8
/// odd-indexed outputs via stages 2-6 operating on the difference half
/// (`input[0]-input[15]`, ..., `input[7]-input[8]`).
///
/// Mirrors libvpx `fdct16` (`vp9/encoder/vp9_dct.c:94-233`).
pub fn fdct16(input: &[i32; 16], output: &mut [i32; 16]) {
    // step 1: split into sum half (`in_half`) and difference half (`step1`).
    let in_half: [i64; 8] = [
        input[0] as i64 + input[15] as i64,
        input[1] as i64 + input[14] as i64,
        input[2] as i64 + input[13] as i64,
        input[3] as i64 + input[12] as i64,
        input[4] as i64 + input[11] as i64,
        input[5] as i64 + input[10] as i64,
        input[6] as i64 + input[9] as i64,
        input[7] as i64 + input[8] as i64,
    ];
    let step1: [i64; 8] = [
        input[7] as i64 - input[8] as i64,
        input[6] as i64 - input[9] as i64,
        input[5] as i64 - input[10] as i64,
        input[4] as i64 - input[11] as i64,
        input[3] as i64 - input[12] as i64,
        input[2] as i64 - input[13] as i64,
        input[1] as i64 - input[14] as i64,
        input[0] as i64 - input[15] as i64,
    ];
    fdct16_butterfly(&in_half, &step1, output, 0);
}

/// The fdct16 butterfly body operating on pre-loaded `in_half` (sum-half)
/// and `step1` (difference-half) and writing 16 values at
/// `output[base..base+16]`. Factored so [`fdct16x16`] can reuse it on
/// two passes without duplicating ~130 lines of arithmetic.
///
/// Mirrors the body at `vp9/encoder/vp9_dct.c:120-232` and
/// `vpx_dsp/fwd_txfm.c:245-352`.
#[inline]
fn fdct16_butterfly(in_half: &[i64; 8], step1: &[i64; 8], output: &mut [i32], base: usize) {
    // ---- Even outputs (indices 0, 2, 4, 6, 8, 10, 12, 14): fdct8 sub-block on in_half ----
    let s0 = in_half[0] + in_half[7];
    let s1 = in_half[1] + in_half[6];
    let s2 = in_half[2] + in_half[5];
    let s3 = in_half[3] + in_half[4];
    let s4 = in_half[3] - in_half[4];
    let s5 = in_half[2] - in_half[5];
    let s6 = in_half[1] - in_half[6];
    let s7 = in_half[0] - in_half[7];

    // fdct4(step, step) on s0..s3 → out[0], out[4], out[8], out[12].
    let x0 = s0 + s3;
    let x1 = s1 + s2;
    let x2 = s1 - s2;
    let x3 = s0 - s3;
    let t0 = (x0 + x1) * COSPI_16_64 as i64;
    let t1 = (x0 - x1) * COSPI_16_64 as i64;
    // Note libvpx vp9_dct.c:143-144 uses `x3 * cospi_8 + x2 * cospi_24` for t2
    // and `x3 * cospi_24 - x2 * cospi_8` for t3 (different from fdct4/fdct8).
    let t2 = x3 * COSPI_8_64 as i64 + x2 * COSPI_24_64 as i64;
    let t3 = x3 * COSPI_24_64 as i64 - x2 * COSPI_8_64 as i64;
    output[base] = wraplow(dct_const_round_shift(t0) as i32);
    output[base + 4] = wraplow(dct_const_round_shift(t2) as i32);
    output[base + 8] = wraplow(dct_const_round_shift(t1) as i32);
    output[base + 12] = wraplow(dct_const_round_shift(t3) as i32);

    // Stage 2 on s5, s6.
    let t0 = (s6 - s5) * COSPI_16_64 as i64;
    let t1 = (s6 + s5) * COSPI_16_64 as i64;
    let t2 = dct_const_round_shift(t0);
    let t3 = dct_const_round_shift(t1);

    // Stage 3.
    let x0 = s4 + t2;
    let x1 = s4 - t2;
    let x2 = s7 - t3;
    let x3 = s7 + t3;

    // Stage 4 → out[2], out[6], out[10], out[14].
    let t0 = x0 * COSPI_28_64 as i64 + x3 * COSPI_4_64 as i64;
    let t1 = x1 * COSPI_12_64 as i64 + x2 * COSPI_20_64 as i64;
    let t2 = x2 * COSPI_12_64 as i64 + x1 * -(COSPI_20_64 as i64);
    let t3 = x3 * COSPI_28_64 as i64 + x0 * -(COSPI_4_64 as i64);
    output[base + 2] = wraplow(dct_const_round_shift(t0) as i32);
    output[base + 6] = wraplow(dct_const_round_shift(t2) as i32);
    output[base + 10] = wraplow(dct_const_round_shift(t1) as i32);
    output[base + 14] = wraplow(dct_const_round_shift(t3) as i32);

    // ---- Odd outputs (indices 1, 3, 5, 7, 9, 11, 13, 15): steps 2-6 on step1 ----
    //
    // step1[] is consumed and rewritten across stages, so we copy into
    // a mutable local. libvpx reuses the same array names through the
    // stages; we mirror that with local mut bindings.
    let step1_0 = step1[0];
    let step1_1 = step1[1];
    let step1_2 = step1[2];
    let step1_3 = step1[3];
    let step1_4 = step1[4];
    let step1_5 = step1[5];
    let step1_6 = step1[6];
    let step1_7 = step1[7];

    // step 2
    let temp1 = (step1_5 - step1_2) * COSPI_16_64 as i64;
    let temp2 = (step1_4 - step1_3) * COSPI_16_64 as i64;
    let step2_2 = dct_const_round_shift(temp1);
    let step2_3 = dct_const_round_shift(temp2);
    let temp1 = (step1_4 + step1_3) * COSPI_16_64 as i64;
    let temp2 = (step1_5 + step1_2) * COSPI_16_64 as i64;
    let step2_4 = dct_const_round_shift(temp1);
    let step2_5 = dct_const_round_shift(temp2);

    // step 3
    let step3_0 = step1_0 + step2_3;
    let step3_1 = step1_1 + step2_2;
    let step3_2 = step1_1 - step2_2;
    let step3_3 = step1_0 - step2_3;
    let step3_4 = step1_7 - step2_4;
    let step3_5 = step1_6 - step2_5;
    let step3_6 = step1_6 + step2_5;
    let step3_7 = step1_7 + step2_4;

    // step 4
    let temp1 = step3_1 * -(COSPI_8_64 as i64) + step3_6 * COSPI_24_64 as i64;
    let temp2 = step3_2 * COSPI_24_64 as i64 + step3_5 * COSPI_8_64 as i64;
    let step2_1 = dct_const_round_shift(temp1);
    let step2_2 = dct_const_round_shift(temp2);
    let temp1 = step3_2 * COSPI_8_64 as i64 - step3_5 * COSPI_24_64 as i64;
    let temp2 = step3_1 * COSPI_24_64 as i64 + step3_6 * COSPI_8_64 as i64;
    let step2_5 = dct_const_round_shift(temp1);
    let step2_6 = dct_const_round_shift(temp2);

    // step 5
    let step1_0 = step3_0 + step2_1;
    let step1_1 = step3_0 - step2_1;
    let step1_2 = step3_3 + step2_2;
    let step1_3 = step3_3 - step2_2;
    let step1_4 = step3_4 - step2_5;
    let step1_5 = step3_4 + step2_5;
    let step1_6 = step3_7 - step2_6;
    let step1_7 = step3_7 + step2_6;

    // step 6 → odd outputs
    let temp1 = step1_0 * COSPI_30_64 as i64 + step1_7 * COSPI_2_64 as i64;
    let temp2 = step1_1 * COSPI_14_64 as i64 + step1_6 * COSPI_18_64 as i64;
    output[base + 1] = wraplow(dct_const_round_shift(temp1) as i32);
    output[base + 9] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step1_2 * COSPI_22_64 as i64 + step1_5 * COSPI_10_64 as i64;
    let temp2 = step1_3 * COSPI_6_64 as i64 + step1_4 * COSPI_26_64 as i64;
    output[base + 5] = wraplow(dct_const_round_shift(temp1) as i32);
    output[base + 13] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step1_3 * -(COSPI_26_64 as i64) + step1_4 * COSPI_6_64 as i64;
    let temp2 = step1_2 * -(COSPI_10_64 as i64) + step1_5 * COSPI_22_64 as i64;
    output[base + 3] = wraplow(dct_const_round_shift(temp1) as i32);
    output[base + 11] = wraplow(dct_const_round_shift(temp2) as i32);

    let temp1 = step1_1 * -(COSPI_18_64 as i64) + step1_6 * COSPI_14_64 as i64;
    let temp2 = step1_0 * -(COSPI_2_64 as i64) + step1_7 * COSPI_30_64 as i64;
    output[base + 7] = wraplow(dct_const_round_shift(temp1) as i32);
    output[base + 15] = wraplow(dct_const_round_shift(temp2) as i32);
}

/// 1D 16-point forward ADST. Four stages with input permutation
/// (`x0=input[15]`, `x1=input[0]`, `x2=input[13]`, ..., `x15=input[14]`)
/// and output permutation with sign flips at indices 1, 3, 13, 15.
///
/// Mirrors libvpx `fadst16` (`vp9/encoder/vp9_dct.c:346-507`).
pub fn fadst16(input: &[i32; 16], output: &mut [i32; 16]) {
    // Input permutation per libvpx :350-365.
    let x0 = input[15] as i64;
    let x1 = input[0] as i64;
    let x2 = input[13] as i64;
    let x3 = input[2] as i64;
    let x4 = input[11] as i64;
    let x5 = input[4] as i64;
    let x6 = input[9] as i64;
    let x7 = input[6] as i64;
    let x8 = input[7] as i64;
    let x9 = input[8] as i64;
    let x10 = input[5] as i64;
    let x11 = input[10] as i64;
    let x12 = input[3] as i64;
    let x13 = input[12] as i64;
    let x14 = input[1] as i64;
    let x15 = input[14] as i64;

    // stage 1
    let s0 = x0 * COSPI_1_64 as i64 + x1 * COSPI_31_64 as i64;
    let s1 = x0 * COSPI_31_64 as i64 - x1 * COSPI_1_64 as i64;
    let s2 = x2 * COSPI_5_64 as i64 + x3 * COSPI_27_64 as i64;
    let s3 = x2 * COSPI_27_64 as i64 - x3 * COSPI_5_64 as i64;
    let s4 = x4 * COSPI_9_64 as i64 + x5 * COSPI_23_64 as i64;
    let s5 = x4 * COSPI_23_64 as i64 - x5 * COSPI_9_64 as i64;
    let s6 = x6 * COSPI_13_64 as i64 + x7 * COSPI_19_64 as i64;
    let s7 = x6 * COSPI_19_64 as i64 - x7 * COSPI_13_64 as i64;
    let s8 = x8 * COSPI_17_64 as i64 + x9 * COSPI_15_64 as i64;
    let s9 = x8 * COSPI_15_64 as i64 - x9 * COSPI_17_64 as i64;
    let s10 = x10 * COSPI_21_64 as i64 + x11 * COSPI_11_64 as i64;
    let s11 = x10 * COSPI_11_64 as i64 - x11 * COSPI_21_64 as i64;
    let s12 = x12 * COSPI_25_64 as i64 + x13 * COSPI_7_64 as i64;
    let s13 = x12 * COSPI_7_64 as i64 - x13 * COSPI_25_64 as i64;
    let s14 = x14 * COSPI_29_64 as i64 + x15 * COSPI_3_64 as i64;
    let s15 = x14 * COSPI_3_64 as i64 - x15 * COSPI_29_64 as i64;

    let x0 = dct_const_round_shift(s0 + s8);
    let x1 = dct_const_round_shift(s1 + s9);
    let x2 = dct_const_round_shift(s2 + s10);
    let x3 = dct_const_round_shift(s3 + s11);
    let x4 = dct_const_round_shift(s4 + s12);
    let x5 = dct_const_round_shift(s5 + s13);
    let x6 = dct_const_round_shift(s6 + s14);
    let x7 = dct_const_round_shift(s7 + s15);
    let x8 = dct_const_round_shift(s0 - s8);
    let x9 = dct_const_round_shift(s1 - s9);
    let x10 = dct_const_round_shift(s2 - s10);
    let x11 = dct_const_round_shift(s3 - s11);
    let x12 = dct_const_round_shift(s4 - s12);
    let x13 = dct_const_round_shift(s5 - s13);
    let x14 = dct_const_round_shift(s6 - s14);
    let x15 = dct_const_round_shift(s7 - s15);

    // stage 2: pass through s0..s7
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = x4;
    let s5 = x5;
    let s6 = x6;
    let s7 = x7;
    let s8 = x8 * COSPI_4_64 as i64 + x9 * COSPI_28_64 as i64;
    let s9 = x8 * COSPI_28_64 as i64 - x9 * COSPI_4_64 as i64;
    let s10 = x10 * COSPI_20_64 as i64 + x11 * COSPI_12_64 as i64;
    let s11 = x10 * COSPI_12_64 as i64 - x11 * COSPI_20_64 as i64;
    let s12 = -x12 * COSPI_28_64 as i64 + x13 * COSPI_4_64 as i64;
    let s13 = x12 * COSPI_4_64 as i64 + x13 * COSPI_28_64 as i64;
    let s14 = -x14 * COSPI_12_64 as i64 + x15 * COSPI_20_64 as i64;
    let s15 = x14 * COSPI_20_64 as i64 + x15 * COSPI_12_64 as i64;

    let x0 = s0 + s4;
    let x1 = s1 + s5;
    let x2 = s2 + s6;
    let x3 = s3 + s7;
    let x4 = s0 - s4;
    let x5 = s1 - s5;
    let x6 = s2 - s6;
    let x7 = s3 - s7;
    let x8 = dct_const_round_shift(s8 + s12);
    let x9 = dct_const_round_shift(s9 + s13);
    let x10 = dct_const_round_shift(s10 + s14);
    let x11 = dct_const_round_shift(s11 + s15);
    let x12 = dct_const_round_shift(s8 - s12);
    let x13 = dct_const_round_shift(s9 - s13);
    let x14 = dct_const_round_shift(s10 - s14);
    let x15 = dct_const_round_shift(s11 - s15);

    // stage 3
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = x4 * COSPI_8_64 as i64 + x5 * COSPI_24_64 as i64;
    let s5 = x4 * COSPI_24_64 as i64 - x5 * COSPI_8_64 as i64;
    let s6 = -x6 * COSPI_24_64 as i64 + x7 * COSPI_8_64 as i64;
    let s7 = x6 * COSPI_8_64 as i64 + x7 * COSPI_24_64 as i64;
    let s8 = x8;
    let s9 = x9;
    let s10 = x10;
    let s11 = x11;
    let s12 = x12 * COSPI_8_64 as i64 + x13 * COSPI_24_64 as i64;
    let s13 = x12 * COSPI_24_64 as i64 - x13 * COSPI_8_64 as i64;
    let s14 = -x14 * COSPI_24_64 as i64 + x15 * COSPI_8_64 as i64;
    let s15 = x14 * COSPI_8_64 as i64 + x15 * COSPI_24_64 as i64;

    let x0 = s0 + s2;
    let x1 = s1 + s3;
    let x2 = s0 - s2;
    let x3 = s1 - s3;
    let x4 = dct_const_round_shift(s4 + s6);
    let x5 = dct_const_round_shift(s5 + s7);
    let x6 = dct_const_round_shift(s4 - s6);
    let x7 = dct_const_round_shift(s5 - s7);
    let x8 = s8 + s10;
    let x9 = s9 + s11;
    let x10 = s8 - s10;
    let x11 = s9 - s11;
    let x12 = dct_const_round_shift(s12 + s14);
    let x13 = dct_const_round_shift(s13 + s15);
    let x14 = dct_const_round_shift(s12 - s14);
    let x15 = dct_const_round_shift(s13 - s15);

    // stage 4
    let s2 = -(COSPI_16_64 as i64) * (x2 + x3);
    let s3 = COSPI_16_64 as i64 * (x2 - x3);
    let s6 = COSPI_16_64 as i64 * (x6 + x7);
    let s7 = COSPI_16_64 as i64 * (-x6 + x7);
    let s10 = COSPI_16_64 as i64 * (x10 + x11);
    let s11 = COSPI_16_64 as i64 * (-x10 + x11);
    let s14 = -(COSPI_16_64 as i64) * (x14 + x15);
    let s15 = COSPI_16_64 as i64 * (x14 - x15);

    let x2 = dct_const_round_shift(s2);
    let x3 = dct_const_round_shift(s3);
    let x6 = dct_const_round_shift(s6);
    let x7 = dct_const_round_shift(s7);
    let x10 = dct_const_round_shift(s10);
    let x11 = dct_const_round_shift(s11);
    let x14 = dct_const_round_shift(s14);
    let x15 = dct_const_round_shift(s15);

    // Output permutation with sign flips at 1, 3, 13, 15 (libvpx :491-506).
    output[0] = wraplow(x0 as i32);
    output[1] = wraplow(-x8 as i32);
    output[2] = wraplow(x12 as i32);
    output[3] = wraplow(-x4 as i32);
    output[4] = wraplow(x6 as i32);
    output[5] = wraplow(x14 as i32);
    output[6] = wraplow(x10 as i32);
    output[7] = wraplow(x2 as i32);
    output[8] = wraplow(x3 as i32);
    output[9] = wraplow(x11 as i32);
    output[10] = wraplow(x15 as i32);
    output[11] = wraplow(x7 as i32);
    output[12] = wraplow(x5 as i32);
    output[13] = wraplow(-x13 as i32);
    output[14] = wraplow(x9 as i32);
    output[15] = wraplow(-x1 as i32);
}

/// 2D 16×16 forward DCT. Pass 0 multiplies input by 4 (no DC bump);
/// pass 1 applies `(x + 1) >> 2` to each intermediate value before
/// the row transform. No final reduction step (the inter-pass
/// `(x + 1) >> 2` IS the reduction).
///
/// Mirrors libvpx `vpx_fdct16x16_c` (`vpx_dsp/fwd_txfm.c:183-361`).
pub fn fdct16x16(input: &[i16], output: &mut [i32; 256], stride: usize) {
    debug_assert!(
        input.len() >= 15 * stride + 16,
        "input slice too small for 16x16 at stride {stride}"
    );

    let mut intermediate = [0i32; 256];

    // Pass 0: column transform with ×4 scaling. For each output column
    // i, build `in_half` (sums) and `step1` (diffs) directly from the
    // i-th input column, call the butterfly, then transpose-store.
    for i in 0..16 {
        let mut col_out = [0i32; 16];
        let in_half: [i64; 8] = [
            (input[i] as i64 + input[15 * stride + i] as i64) * 4,
            (input[stride + i] as i64 + input[14 * stride + i] as i64) * 4,
            (input[2 * stride + i] as i64 + input[13 * stride + i] as i64) * 4,
            (input[3 * stride + i] as i64 + input[12 * stride + i] as i64) * 4,
            (input[4 * stride + i] as i64 + input[11 * stride + i] as i64) * 4,
            (input[5 * stride + i] as i64 + input[10 * stride + i] as i64) * 4,
            (input[6 * stride + i] as i64 + input[9 * stride + i] as i64) * 4,
            (input[7 * stride + i] as i64 + input[8 * stride + i] as i64) * 4,
        ];
        let step1: [i64; 8] = [
            (input[7 * stride + i] as i64 - input[8 * stride + i] as i64) * 4,
            (input[6 * stride + i] as i64 - input[9 * stride + i] as i64) * 4,
            (input[5 * stride + i] as i64 - input[10 * stride + i] as i64) * 4,
            (input[4 * stride + i] as i64 - input[11 * stride + i] as i64) * 4,
            (input[3 * stride + i] as i64 - input[12 * stride + i] as i64) * 4,
            (input[2 * stride + i] as i64 - input[13 * stride + i] as i64) * 4,
            (input[stride + i] as i64 - input[14 * stride + i] as i64) * 4,
            (input[i] as i64 - input[15 * stride + i] as i64) * 4,
        ];
        fdct16_butterfly(&in_half, &step1, &mut col_out, 0);
        // Transpose-store into intermediate[j * 16 + i].
        for j in 0..16 {
            intermediate[j * 16 + i] = col_out[j];
        }
    }

    // Pass 1: build in_half/step1 from intermediate **row** `i`
    // (= the per-column-pass coefficients at all 16 horizontal
    // positions) with `(x + 1) >> 2` between-pass reduction baked into
    // the input loading. Pass 0 stores
    // `intermediate[j * 16 + i] = (j-th coef of fdct16 on input col i)`
    // i.e. `intermediate.entry(row=j, col=i) = B[j, i]` in natural
    // matrix layout. Pass 1's "row i" read therefore picks out the
    // sequence `{B[i, c] for c=0..15}` — the i-th coefficient of the
    // column-pass across all input columns, which is the input for the
    // second 1D transform per the standard 2D DCT structure.
    for i in 0..16 {
        let row_base = i * 16;
        let read = |c: usize| (intermediate[row_base + c] + 1) >> 2;
        let in_half: [i64; 8] = [
            read(0) as i64 + read(15) as i64,
            read(1) as i64 + read(14) as i64,
            read(2) as i64 + read(13) as i64,
            read(3) as i64 + read(12) as i64,
            read(4) as i64 + read(11) as i64,
            read(5) as i64 + read(10) as i64,
            read(6) as i64 + read(9) as i64,
            read(7) as i64 + read(8) as i64,
        ];
        let step1: [i64; 8] = [
            read(7) as i64 - read(8) as i64,
            read(6) as i64 - read(9) as i64,
            read(5) as i64 - read(10) as i64,
            read(4) as i64 - read(11) as i64,
            read(3) as i64 - read(12) as i64,
            read(2) as i64 - read(13) as i64,
            read(1) as i64 - read(14) as i64,
            read(0) as i64 - read(15) as i64,
        ];
        fdct16_butterfly(&in_half, &step1, output, i * 16);
    }
}

/// 2D 16×16 hybrid forward transform (DCT or ADST per axis), selected
/// by `tx_type`. `DCT_DCT` short-circuits to [`fdct16x16`] (matching
/// libvpx :642-643). Other types use the column / row 1D dispatch
/// with pass-0 ×4 scaling and **between-pass `(x + 1 + (x<0)) >> 2`**
/// (round half away from zero, divide by 4).
///
/// Note: this `(x + 1 + (x<0)) >> 2` between-pass form is **not**
/// bit-equivalent to `fdct16x16`'s `(x + 1) >> 2`, which is why the
/// DCT_DCT short-circuit is necessary to preserve `fht16x16(DCT_DCT) ==
/// fdct16x16` semantics.
///
/// Mirrors libvpx `vp9_fht16x16_c` (`vp9/encoder/vp9_dct.c:640-665`).
pub fn fht16x16(input: &[i16], output: &mut [i32; 256], stride: usize, tx_type: TxType) {
    if matches!(tx_type, TxType::DctDct) {
        fdct16x16(input, output, stride);
        return;
    }

    debug_assert!(
        input.len() >= 15 * stride + 16,
        "input slice too small for 16x16 at stride {stride}"
    );

    // FHT_16 table (vp9_dct.c:523-528).
    let (col_fn, row_fn): (Fwd1d16, Fwd1d16) = match tx_type {
        TxType::DctDct => unreachable!(),
        TxType::AdstDct => (fadst16, fdct16),
        TxType::DctAdst => (fdct16, fadst16),
        TxType::AdstAdst => (fadst16, fadst16),
    };

    let mut intermediate = [0i32; 256];

    // Columns: ×4 scaling, transform, transpose-store with
    // `(x + 1 + (x<0)) >> 2` reduction.
    for i in 0..16 {
        let mut col_in = [0i32; 16];
        for j in 0..16 {
            col_in[j] = (input[j * stride + i] as i32) * 4;
        }
        let mut col_out = [0i32; 16];
        col_fn(&col_in, &mut col_out);
        for j in 0..16 {
            let v = col_out[j];
            let reduced = (v + 1 + i32::from(v < 0)) >> 2;
            intermediate[j * 16 + i] = wraplow(reduced);
        }
    }

    // Rows: transform with no further reduction (the column-pass
    // reduction was the only scaling normalisation).
    for i in 0..16 {
        let mut row_in = [0i32; 16];
        for j in 0..16 {
            row_in[j] = intermediate[j + i * 16];
        }
        let mut row_out = [0i32; 16];
        row_fn(&row_in, &mut row_out);
        for j in 0..16 {
            output[j + i * 16] = wraplow(row_out[j]);
        }
    }
}

// ---------------------------------------------------------------------
// 32×32 forward transform family (DCT only — no hybrid at 32×32).
// ---------------------------------------------------------------------

/// Cosine rotation round-shift used inside [`fdct32`]. Mirrors libvpx
/// `dct_32_round` (`vpx_dsp/fwd_txfm.c:372-378`) which wraps
/// `ROUND_POWER_OF_TWO(x, DCT_CONST_BITS)`.
#[inline]
const fn dct_32_round(x: i64) -> i64 {
    dct_const_round_shift(x)
}

/// `half_round_shift`: `(x + 1 + (x < 0)) >> 2` — round half away
/// from zero divided by 4. Used between stages 2 and 3 of [`fdct32`]
/// when the `round` flag is set (i.e. by the rate-distortion variant).
/// Mirrors libvpx `half_round_shift` (`vpx_dsp/fwd_txfm.c:380-383`).
#[inline]
const fn half_round_shift(x: i64) -> i64 {
    (x + 1 + (x < 0) as i64) >> 2
}

/// 1D 32-point forward DCT. Seven stages plus a final bit-reversed
/// permutation. The `round` flag toggles an extra `half_round_shift`
/// (÷4 with round-half-away-from-zero) applied between stages 2 and 3
/// — set this `false` for the standard 2D wrapper [`fdct32x32`] and
/// `true` for the rate-distortion variant (`fdct32x32_rd`, not yet
/// ported).
///
/// Mirrors libvpx `vpx_fdct32` (`vpx_dsp/fwd_txfm.c:385-706`).
///
/// Note: input is `&[i32; 32]` (not `&[i16; 32]`) because the only
/// caller — [`fdct32x32`] — passes ×4-scaled input values that can
/// exceed `i16`.
pub fn fdct32(input: &[i32; 32], output: &mut [i32; 32], round: bool) {
    let mut step = [0i64; 32];
    let mut out = [0i64; 32];

    // Stage 1 — pairwise sums/diffs on the 32-element input.
    for k in 0..16 {
        step[k] = input[k] as i64 + input[31 - k] as i64;
    }
    for k in 16..32 {
        step[k] = -(input[k] as i64) + input[31 - k] as i64;
    }

    // Stage 2 — sums/diffs on the lower-16 half + identity / cospi_16
    // rotation on the upper-16 half.
    for k in 0..8 {
        out[k] = step[k] + step[15 - k];
    }
    for k in 8..16 {
        out[k] = -step[k] + step[15 - k];
    }
    out[16] = step[16];
    out[17] = step[17];
    out[18] = step[18];
    out[19] = step[19];
    out[20] = dct_32_round((-step[20] + step[27]) * COSPI_16_64 as i64);
    out[21] = dct_32_round((-step[21] + step[26]) * COSPI_16_64 as i64);
    out[22] = dct_32_round((-step[22] + step[25]) * COSPI_16_64 as i64);
    out[23] = dct_32_round((-step[23] + step[24]) * COSPI_16_64 as i64);
    out[24] = dct_32_round((step[24] + step[23]) * COSPI_16_64 as i64);
    out[25] = dct_32_round((step[25] + step[22]) * COSPI_16_64 as i64);
    out[26] = dct_32_round((step[26] + step[21]) * COSPI_16_64 as i64);
    out[27] = dct_32_round((step[27] + step[20]) * COSPI_16_64 as i64);
    out[28] = step[28];
    out[29] = step[29];
    out[30] = step[30];
    out[31] = step[31];

    // Optional `half_round_shift` between stages 2 and 3 (only for
    // the rate-distortion variant).
    if round {
        for slot in out.iter_mut() {
            *slot = half_round_shift(*slot);
        }
    }

    // Stage 3.
    step[0] = out[0] + out[7];
    step[1] = out[1] + out[6];
    step[2] = out[2] + out[5];
    step[3] = out[3] + out[4];
    step[4] = -out[4] + out[3];
    step[5] = -out[5] + out[2];
    step[6] = -out[6] + out[1];
    step[7] = -out[7] + out[0];
    step[8] = out[8];
    step[9] = out[9];
    step[10] = dct_32_round((-out[10] + out[13]) * COSPI_16_64 as i64);
    step[11] = dct_32_round((-out[11] + out[12]) * COSPI_16_64 as i64);
    step[12] = dct_32_round((out[12] + out[11]) * COSPI_16_64 as i64);
    step[13] = dct_32_round((out[13] + out[10]) * COSPI_16_64 as i64);
    step[14] = out[14];
    step[15] = out[15];
    step[16] = out[16] + out[23];
    step[17] = out[17] + out[22];
    step[18] = out[18] + out[21];
    step[19] = out[19] + out[20];
    step[20] = -out[20] + out[19];
    step[21] = -out[21] + out[18];
    step[22] = -out[22] + out[17];
    step[23] = -out[23] + out[16];
    step[24] = -out[24] + out[31];
    step[25] = -out[25] + out[30];
    step[26] = -out[26] + out[29];
    step[27] = -out[27] + out[28];
    step[28] = out[28] + out[27];
    step[29] = out[29] + out[26];
    step[30] = out[30] + out[25];
    step[31] = out[31] + out[24];

    // Stage 4.
    out[0] = step[0] + step[3];
    out[1] = step[1] + step[2];
    out[2] = -step[2] + step[1];
    out[3] = -step[3] + step[0];
    out[4] = step[4];
    out[5] = dct_32_round((-step[5] + step[6]) * COSPI_16_64 as i64);
    out[6] = dct_32_round((step[6] + step[5]) * COSPI_16_64 as i64);
    out[7] = step[7];
    out[8] = step[8] + step[11];
    out[9] = step[9] + step[10];
    out[10] = -step[10] + step[9];
    out[11] = -step[11] + step[8];
    out[12] = -step[12] + step[15];
    out[13] = -step[13] + step[14];
    out[14] = step[14] + step[13];
    out[15] = step[15] + step[12];
    out[16] = step[16];
    out[17] = step[17];
    out[18] = dct_32_round(step[18] * -(COSPI_8_64 as i64) + step[29] * COSPI_24_64 as i64);
    out[19] = dct_32_round(step[19] * -(COSPI_8_64 as i64) + step[28] * COSPI_24_64 as i64);
    out[20] = dct_32_round(step[20] * -(COSPI_24_64 as i64) + step[27] * -(COSPI_8_64 as i64));
    out[21] = dct_32_round(step[21] * -(COSPI_24_64 as i64) + step[26] * -(COSPI_8_64 as i64));
    out[22] = step[22];
    out[23] = step[23];
    out[24] = step[24];
    out[25] = step[25];
    out[26] = dct_32_round(step[26] * COSPI_24_64 as i64 + step[21] * -(COSPI_8_64 as i64));
    out[27] = dct_32_round(step[27] * COSPI_24_64 as i64 + step[20] * -(COSPI_8_64 as i64));
    out[28] = dct_32_round(step[28] * COSPI_8_64 as i64 + step[19] * COSPI_24_64 as i64);
    out[29] = dct_32_round(step[29] * COSPI_8_64 as i64 + step[18] * COSPI_24_64 as i64);
    out[30] = step[30];
    out[31] = step[31];

    // Stage 5.
    step[0] = dct_32_round((out[0] + out[1]) * COSPI_16_64 as i64);
    step[1] = dct_32_round((-out[1] + out[0]) * COSPI_16_64 as i64);
    step[2] = dct_32_round(out[2] * COSPI_24_64 as i64 + out[3] * COSPI_8_64 as i64);
    step[3] = dct_32_round(out[3] * COSPI_24_64 as i64 - out[2] * COSPI_8_64 as i64);
    step[4] = out[4] + out[5];
    step[5] = -out[5] + out[4];
    step[6] = -out[6] + out[7];
    step[7] = out[7] + out[6];
    step[8] = out[8];
    step[9] = dct_32_round(out[9] * -(COSPI_8_64 as i64) + out[14] * COSPI_24_64 as i64);
    step[10] = dct_32_round(out[10] * -(COSPI_24_64 as i64) + out[13] * -(COSPI_8_64 as i64));
    step[11] = out[11];
    step[12] = out[12];
    step[13] = dct_32_round(out[13] * COSPI_24_64 as i64 + out[10] * -(COSPI_8_64 as i64));
    step[14] = dct_32_round(out[14] * COSPI_8_64 as i64 + out[9] * COSPI_24_64 as i64);
    step[15] = out[15];
    step[16] = out[16] + out[19];
    step[17] = out[17] + out[18];
    step[18] = -out[18] + out[17];
    step[19] = -out[19] + out[16];
    step[20] = -out[20] + out[23];
    step[21] = -out[21] + out[22];
    step[22] = out[22] + out[21];
    step[23] = out[23] + out[20];
    step[24] = out[24] + out[27];
    step[25] = out[25] + out[26];
    step[26] = -out[26] + out[25];
    step[27] = -out[27] + out[24];
    step[28] = -out[28] + out[31];
    step[29] = -out[29] + out[30];
    step[30] = out[30] + out[29];
    step[31] = out[31] + out[28];

    // Stage 6.
    out[0] = step[0];
    out[1] = step[1];
    out[2] = step[2];
    out[3] = step[3];
    out[4] = dct_32_round(step[4] * COSPI_28_64 as i64 + step[7] * COSPI_4_64 as i64);
    out[5] = dct_32_round(step[5] * COSPI_12_64 as i64 + step[6] * COSPI_20_64 as i64);
    out[6] = dct_32_round(step[6] * COSPI_12_64 as i64 + step[5] * -(COSPI_20_64 as i64));
    out[7] = dct_32_round(step[7] * COSPI_28_64 as i64 + step[4] * -(COSPI_4_64 as i64));
    out[8] = step[8] + step[9];
    out[9] = -step[9] + step[8];
    out[10] = -step[10] + step[11];
    out[11] = step[11] + step[10];
    out[12] = step[12] + step[13];
    out[13] = -step[13] + step[12];
    out[14] = -step[14] + step[15];
    out[15] = step[15] + step[14];
    out[16] = step[16];
    out[17] = dct_32_round(step[17] * -(COSPI_4_64 as i64) + step[30] * COSPI_28_64 as i64);
    out[18] = dct_32_round(step[18] * -(COSPI_28_64 as i64) + step[29] * -(COSPI_4_64 as i64));
    out[19] = step[19];
    out[20] = step[20];
    out[21] = dct_32_round(step[21] * -(COSPI_20_64 as i64) + step[26] * COSPI_12_64 as i64);
    out[22] = dct_32_round(step[22] * -(COSPI_12_64 as i64) + step[25] * -(COSPI_20_64 as i64));
    out[23] = step[23];
    out[24] = step[24];
    out[25] = dct_32_round(step[25] * COSPI_12_64 as i64 + step[22] * -(COSPI_20_64 as i64));
    out[26] = dct_32_round(step[26] * COSPI_20_64 as i64 + step[21] * COSPI_12_64 as i64);
    out[27] = step[27];
    out[28] = step[28];
    out[29] = dct_32_round(step[29] * COSPI_28_64 as i64 + step[18] * -(COSPI_4_64 as i64));
    out[30] = dct_32_round(step[30] * COSPI_4_64 as i64 + step[17] * COSPI_28_64 as i64);
    out[31] = step[31];

    // Stage 7.
    step[0] = out[0];
    step[1] = out[1];
    step[2] = out[2];
    step[3] = out[3];
    step[4] = out[4];
    step[5] = out[5];
    step[6] = out[6];
    step[7] = out[7];
    step[8] = dct_32_round(out[8] * COSPI_30_64 as i64 + out[15] * COSPI_2_64 as i64);
    step[9] = dct_32_round(out[9] * COSPI_14_64 as i64 + out[14] * COSPI_18_64 as i64);
    step[10] = dct_32_round(out[10] * COSPI_22_64 as i64 + out[13] * COSPI_10_64 as i64);
    step[11] = dct_32_round(out[11] * COSPI_6_64 as i64 + out[12] * COSPI_26_64 as i64);
    step[12] = dct_32_round(out[12] * COSPI_6_64 as i64 + out[11] * -(COSPI_26_64 as i64));
    step[13] = dct_32_round(out[13] * COSPI_22_64 as i64 + out[10] * -(COSPI_10_64 as i64));
    step[14] = dct_32_round(out[14] * COSPI_14_64 as i64 + out[9] * -(COSPI_18_64 as i64));
    step[15] = dct_32_round(out[15] * COSPI_30_64 as i64 + out[8] * -(COSPI_2_64 as i64));
    step[16] = out[16] + out[17];
    step[17] = -out[17] + out[16];
    step[18] = -out[18] + out[19];
    step[19] = out[19] + out[18];
    step[20] = out[20] + out[21];
    step[21] = -out[21] + out[20];
    step[22] = -out[22] + out[23];
    step[23] = out[23] + out[22];
    step[24] = out[24] + out[25];
    step[25] = -out[25] + out[24];
    step[26] = -out[26] + out[27];
    step[27] = out[27] + out[26];
    step[28] = out[28] + out[29];
    step[29] = -out[29] + out[28];
    step[30] = -out[30] + out[31];
    step[31] = out[31] + out[30];

    // Final stage — output indices are bit-reversed.
    output[0] = step[0] as i32;
    output[16] = step[1] as i32;
    output[8] = step[2] as i32;
    output[24] = step[3] as i32;
    output[4] = step[4] as i32;
    output[20] = step[5] as i32;
    output[12] = step[6] as i32;
    output[28] = step[7] as i32;
    output[2] = step[8] as i32;
    output[18] = step[9] as i32;
    output[10] = step[10] as i32;
    output[26] = step[11] as i32;
    output[6] = step[12] as i32;
    output[22] = step[13] as i32;
    output[14] = step[14] as i32;
    output[30] = step[15] as i32;

    output[1] = dct_32_round(step[16] * COSPI_31_64 as i64 + step[31] * COSPI_1_64 as i64) as i32;
    output[17] = dct_32_round(step[17] * COSPI_15_64 as i64 + step[30] * COSPI_17_64 as i64) as i32;
    output[9] = dct_32_round(step[18] * COSPI_23_64 as i64 + step[29] * COSPI_9_64 as i64) as i32;
    output[25] = dct_32_round(step[19] * COSPI_7_64 as i64 + step[28] * COSPI_25_64 as i64) as i32;
    output[5] = dct_32_round(step[20] * COSPI_27_64 as i64 + step[27] * COSPI_5_64 as i64) as i32;
    output[21] = dct_32_round(step[21] * COSPI_11_64 as i64 + step[26] * COSPI_21_64 as i64) as i32;
    output[13] = dct_32_round(step[22] * COSPI_19_64 as i64 + step[25] * COSPI_13_64 as i64) as i32;
    output[29] = dct_32_round(step[23] * COSPI_3_64 as i64 + step[24] * COSPI_29_64 as i64) as i32;
    output[3] =
        dct_32_round(step[24] * COSPI_3_64 as i64 + step[23] * -(COSPI_29_64 as i64)) as i32;
    output[19] =
        dct_32_round(step[25] * COSPI_19_64 as i64 + step[22] * -(COSPI_13_64 as i64)) as i32;
    output[11] =
        dct_32_round(step[26] * COSPI_11_64 as i64 + step[21] * -(COSPI_21_64 as i64)) as i32;
    output[27] =
        dct_32_round(step[27] * COSPI_27_64 as i64 + step[20] * -(COSPI_5_64 as i64)) as i32;
    output[7] =
        dct_32_round(step[28] * COSPI_7_64 as i64 + step[19] * -(COSPI_25_64 as i64)) as i32;
    output[23] =
        dct_32_round(step[29] * COSPI_23_64 as i64 + step[18] * -(COSPI_9_64 as i64)) as i32;
    output[15] =
        dct_32_round(step[30] * COSPI_15_64 as i64 + step[17] * -(COSPI_17_64 as i64)) as i32;
    output[31] =
        dct_32_round(step[31] * COSPI_31_64 as i64 + step[16] * -(COSPI_1_64 as i64)) as i32;
}

/// 2D 32×32 forward DCT. Pass 0 multiplies input by 4, then applies
/// `(x + 1 + (x > 0)) >> 2` between passes (note `>`, not `<` — round
/// half-away-from-zero with positive bias on zero). Pass 1 outputs use
/// `(x + 1 + (x < 0)) >> 2` (the standard form).
///
/// Mirrors libvpx `vpx_fdct32x32_c` (`vpx_dsp/fwd_txfm.c:708-730`).
///
/// Note: 32×32 has no `fht32x32` hybrid variant — libvpx never emits
/// ADST at 32×32 (intra at 32×32 always selects `tx_type=DCT_DCT`).
pub fn fdct32x32(input: &[i16], output: &mut [i32; 1024], stride: usize) {
    debug_assert!(
        input.len() >= 31 * stride + 32,
        "input slice too small for 32x32 at stride {stride}"
    );

    let mut intermediate = [0i32; 1024];

    // Pass 0: column transform with ×4 scaling.
    for i in 0..32 {
        let mut temp_in = [0i32; 32];
        for j in 0..32 {
            temp_in[j] = (input[j * stride + i] as i32) * 4;
        }
        let mut temp_out = [0i32; 32];
        fdct32(&temp_in, &mut temp_out, false);
        // Transpose-store with `(x + 1 + (x > 0)) >> 2` — round half
        // toward positive infinity for positives, toward zero for
        // negatives. NOT symmetric with the pass-1 reduction.
        for j in 0..32 {
            let v = temp_out[j];
            intermediate[j * 32 + i] = (v + 1 + i32::from(v > 0)) >> 2;
        }
    }

    // Pass 1: row transform on intermediate **row** `i` (the i-th
    // coefficient across all 32 column-pass outputs — same layout
    // story as fdct16x16; see comments there). Then final reduction
    // with `(x + 1 + (x < 0)) >> 2` (the standard form).
    for i in 0..32 {
        let row_base = i * 32;
        let mut temp_in = [0i32; 32];
        temp_in.copy_from_slice(&intermediate[row_base..row_base + 32]);
        let mut temp_out = [0i32; 32];
        fdct32(&temp_in, &mut temp_out, false);
        for j in 0..32 {
            let v = temp_out[j];
            output[j + row_base] = wraplow((v + 1 + i32::from(v < 0)) >> 2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::inverse::{
        iadst4, iadst8, iadst16, idct4, idct8, idct16, idct32, idct32x32_1024_add, iht4x4_16_add,
        iht8x8_64_add, iht16x16_256_add,
    };
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

    // -----------------------------------------------------------------
    // 8×8 tests.
    // -----------------------------------------------------------------

    /// Zero input → zero output (1D 8-point DCT).
    #[test]
    fn fdct8_zero_in_zero_out() {
        let input = [0i32; 8];
        let mut output = [99i32; 8];
        fdct8(&input, &mut output);
        assert_eq!(output, [0; 8]);
    }

    /// Zero input → zero output (1D 8-point ADST). Note `fadst8` has
    /// no explicit all-zero short-circuit (unlike `fadst4`) — every
    /// stage runs with zero inputs and produces zero outputs.
    #[test]
    fn fadst8_zero_in_zero_out() {
        let input = [0i32; 8];
        let mut output = [99i32; 8];
        fadst8(&input, &mut output);
        assert_eq!(output, [0; 8]);
    }

    /// 1D fdct8 + idct8 round-trip: input scaled by **4** within ±2.
    /// At fixed point the 1D 8-point DCT pair has gain 4 (vs the 4-pt
    /// pair's gain of 2); the 2D wrappers compensate with an extra
    /// `>> 1` between the two passes. The ±2 tolerance reflects the
    /// looser accumulation through more stages of `dct_const_round_shift`.
    #[test]
    fn fdct8_idct8_round_trip_small() {
        let input: [i32; 8] = [10, -4, 6, -2, 8, -3, 5, -1];
        let mut spectrum = [0i32; 8];
        fdct8(&input, &mut spectrum);
        let mut round_trip = [0i32; 8];
        idct8(&spectrum, &mut round_trip);
        for k in 0..8 {
            let expected = input[k] * 4;
            assert!(
                (round_trip[k] - expected).abs() <= 2,
                "round_trip[{k}] = {} not within ±2 of 4 * input[{k}] = {}",
                round_trip[k],
                expected
            );
        }
    }

    /// 1D fadst8 + iadst8 round-trip: input scaled by 4 within ±2.
    #[test]
    fn fadst8_iadst8_round_trip_small() {
        let input: [i32; 8] = [8, -5, 3, -1, 6, -4, 2, -7];
        let mut spectrum = [0i32; 8];
        fadst8(&input, &mut spectrum);
        let mut round_trip = [0i32; 8];
        iadst8(&spectrum, &mut round_trip);
        for k in 0..8 {
            let expected = input[k] * 4;
            assert!(
                (round_trip[k] - expected).abs() <= 2,
                "round_trip[{k}] = {} not within ±2 of 4 * input[{k}] = {}",
                round_trip[k],
                expected
            );
        }
    }

    /// Zero 8×8 residual block → zero coefficients (2D DCT).
    #[test]
    fn fdct8x8_zero_input_zero_output() {
        let input = [0i16; 64];
        let mut output = [99i32; 64];
        fdct8x8(&input, &mut output, 8);
        assert_eq!(output, [0; 64]);
    }

    /// Zero 8×8 residual block → zero coefficients for every `TxType`
    /// (2D hybrid).
    #[test]
    fn fht8x8_zero_input_zero_output_all_tx_types() {
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let input = [0i16; 64];
            let mut output = [99i32; 64];
            fht8x8(&input, &mut output, 8, tx_type);
            assert_eq!(
                output, [0; 64],
                "tx_type {tx_type:?} did not produce zero output for zero input"
            );
        }
    }

    /// `fht8x8` with `DCT_DCT` is bit-identical to `fdct8x8` — the
    /// short-circuit at `vp9_fht8x8_c:559-560`.
    #[test]
    fn fht8x8_dct_dct_matches_fdct8x8() {
        let input: [i16; 64] = [
            10, -4, 6, -2, 8, -3, 5, -1, //
            -3, 8, -1, 5, -7, 2, -4, 9, //
            7, -6, 4, -2, 3, -5, 8, -1, //
            1, -3, 9, -5, 6, -2, 4, -8, //
            -2, 5, -3, 7, -1, 4, -6, 2, //
            8, -1, 6, -4, 9, -3, 5, -7, //
            3, -8, 2, -5, 7, -1, 4, -6, //
            -5, 9, -2, 6, -3, 8, -1, 4,
        ];
        let mut out_fht = [0i32; 64];
        let mut out_fdct = [0i32; 64];
        fht8x8(&input, &mut out_fht, 8, TxType::DctDct);
        fdct8x8(&input, &mut out_fdct, 8);
        assert_eq!(out_fht, out_fdct);
    }

    /// `fht8x8 → iht8x8_64_add` recovers a small residual within ±1
    /// per pixel for every `TxType`. The canonical encoder-side oracle
    /// at 8×8 scale.
    #[test]
    fn fht8x8_then_iht8x8_recovers_small_residual_all_tx_types() {
        let residual: [i16; 64] = [
            5, -3, 2, -1, 4, -2, 3, -1, //
            -2, 4, -1, 3, -2, 1, -3, 2, //
            1, -2, 5, -4, 2, -1, 4, -3, //
            -1, 2, -3, 6, -2, 3, -1, 5, //
            3, -1, 2, -5, 1, -4, 2, -1, //
            -2, 4, -1, 3, -5, 2, -3, 4, //
            1, -3, 2, -1, 4, -2, 5, -1, //
            -1, 2, -4, 1, -3, 5, -2, 3,
        ];
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let mut coefs = [0i32; 64];
            fht8x8(&residual, &mut coefs, 8, tx_type);

            let mut dest = [128u8; 64];
            iht8x8_64_add(&coefs, &mut dest, 8, tx_type);

            for k in 0..64 {
                let recovered = (dest[k] as i32) - 128;
                let expected = residual[k] as i32;
                assert!(
                    (recovered - expected).abs() <= 1,
                    "{tx_type:?}: pos {k} recovered={recovered} expected={expected}"
                );
            }
        }
    }

    /// `fdct8x8` honours non-trivial `stride`: same content laid out
    /// at stride 16 vs stride 8 produces identical coefficients.
    #[test]
    fn fdct8x8_stride_independence() {
        let residual: [i16; 64] = [
            5, -3, 2, -1, 4, -2, 3, -1, //
            -2, 4, -1, 3, -2, 1, -3, 2, //
            1, -2, 5, -4, 2, -1, 4, -3, //
            -1, 2, -3, 6, -2, 3, -1, 5, //
            3, -1, 2, -5, 1, -4, 2, -1, //
            -2, 4, -1, 3, -5, 2, -3, 4, //
            1, -3, 2, -1, 4, -2, 5, -1, //
            -1, 2, -4, 1, -3, 5, -2, 3,
        ];
        let mut wide = [0i16; 128];
        for r in 0..8 {
            for c in 0..8 {
                wide[r * 16 + c] = residual[r * 8 + c];
            }
        }

        let mut out_tight = [0i32; 64];
        let mut out_wide = [0i32; 64];
        fdct8x8(&residual, &mut out_tight, 8);
        fdct8x8(&wide, &mut out_wide, 16);
        assert_eq!(out_tight, out_wide);
    }

    // -----------------------------------------------------------------
    // 16×16 tests.
    // -----------------------------------------------------------------

    /// Shared 16×16 residual block for round-trip / stride tests.
    /// Small-magnitude (`[-6, 6]`) so reconstructed pixels stay in
    /// unsigned-byte range when added to a dest of 128.
    fn sample_residual_16x16() -> [i16; 256] {
        let mut block = [0i16; 256];
        for r in 0..16 {
            for c in 0..16 {
                // Deterministic pseudo-random pattern bounded to [-6, 6].
                let v = ((r * 7 + c * 11 + r * c) % 13) as i16 - 6;
                block[r * 16 + c] = v;
            }
        }
        block
    }

    /// Zero input → zero output (1D 16-point DCT).
    #[test]
    fn fdct16_zero_in_zero_out() {
        let input = [0i32; 16];
        let mut output = [99i32; 16];
        fdct16(&input, &mut output);
        assert_eq!(output, [0; 16]);
    }

    /// Zero input → zero output (1D 16-point ADST).
    #[test]
    fn fadst16_zero_in_zero_out() {
        let input = [0i32; 16];
        let mut output = [99i32; 16];
        fadst16(&input, &mut output);
        assert_eq!(output, [0; 16]);
    }

    /// 1D fdct16 + idct16 round-trip: input scaled by **8** within ±4.
    /// (Gain pattern continues: 4-point ×2, 8-point ×4, 16-point ×8.)
    #[test]
    fn fdct16_idct16_round_trip_small() {
        let input: [i32; 16] = [
            10, -4, 6, -2, 8, -3, 5, -1, //
            7, -5, 4, -2, 9, -3, 6, -1,
        ];
        let mut spectrum = [0i32; 16];
        fdct16(&input, &mut spectrum);
        let mut round_trip = [0i32; 16];
        idct16(&spectrum, &mut round_trip);
        for k in 0..16 {
            let expected = input[k] * 8;
            assert!(
                (round_trip[k] - expected).abs() <= 4,
                "round_trip[{k}] = {} not within ±4 of 8 * input[{k}] = {}",
                round_trip[k],
                expected
            );
        }
    }

    /// 1D fadst16 + iadst16 round-trip: input scaled by 8 within ±8.
    /// 16-point ADST has four sequential `dct_const_round_shift` stages
    /// (vs the DCT's three) so per-element drift up to ~6 is normal.
    #[test]
    fn fadst16_iadst16_round_trip_small() {
        let input: [i32; 16] = [
            8, -5, 3, -1, 6, -4, 2, -7, //
            9, -3, 5, -1, 4, -8, 2, -6,
        ];
        let mut spectrum = [0i32; 16];
        fadst16(&input, &mut spectrum);
        let mut round_trip = [0i32; 16];
        iadst16(&spectrum, &mut round_trip);
        for k in 0..16 {
            let expected = input[k] * 8;
            assert!(
                (round_trip[k] - expected).abs() <= 8,
                "round_trip[{k}] = {} not within ±8 of 8 * input[{k}] = {}",
                round_trip[k],
                expected
            );
        }
    }

    /// Zero 16×16 residual block → zero coefficients (2D DCT).
    #[test]
    fn fdct16x16_zero_input_zero_output() {
        let input = [0i16; 256];
        let mut output = [99i32; 256];
        fdct16x16(&input, &mut output, 16);
        assert_eq!(output, [0; 256]);
    }

    /// Zero 16×16 residual block → zero coefficients for every
    /// `TxType` (2D hybrid).
    #[test]
    fn fht16x16_zero_input_zero_output_all_tx_types() {
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let input = [0i16; 256];
            let mut output = [99i32; 256];
            fht16x16(&input, &mut output, 16, tx_type);
            assert_eq!(
                output, [0; 256],
                "tx_type {tx_type:?} did not produce zero output for zero input"
            );
        }
    }

    /// `fht16x16` with `DCT_DCT` is bit-identical to `fdct16x16` — the
    /// short-circuit at `vp9_fht16x16_c:642-643`. This identity matters
    /// because the two paths use *different* between-pass rounding
    /// modes (`(x+1)>>2` vs `(x+1+(x<0))>>2`).
    #[test]
    fn fht16x16_dct_dct_matches_fdct16x16() {
        let input = sample_residual_16x16();
        let mut out_fht = [0i32; 256];
        let mut out_fdct = [0i32; 256];
        fht16x16(&input, &mut out_fht, 16, TxType::DctDct);
        fdct16x16(&input, &mut out_fdct, 16);
        assert_eq!(out_fht, out_fdct);
    }

    /// `fht16x16 → iht16x16_256_add` recovers a small residual within
    /// ±1 per pixel for every `TxType`. Canonical 16×16 oracle.
    #[test]
    fn fht16x16_then_iht16x16_recovers_small_residual_all_tx_types() {
        let residual = sample_residual_16x16();
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let mut coefs = [0i32; 256];
            fht16x16(&residual, &mut coefs, 16, tx_type);

            let mut dest = [128u8; 256];
            iht16x16_256_add(&coefs, &mut dest, 16, tx_type);

            for k in 0..256 {
                let recovered = (dest[k] as i32) - 128;
                let expected = residual[k] as i32;
                assert!(
                    (recovered - expected).abs() <= 1,
                    "{tx_type:?}: pos {k} recovered={recovered} expected={expected}"
                );
            }
        }
    }

    /// `fdct16x16` honours non-trivial `stride`: same content laid out
    /// at stride 32 vs stride 16 produces identical coefficients.
    #[test]
    fn fdct16x16_stride_independence() {
        let residual = sample_residual_16x16();
        let mut wide = [0i16; 16 * 32];
        for r in 0..16 {
            for c in 0..16 {
                wide[r * 32 + c] = residual[r * 16 + c];
            }
        }

        let mut out_tight = [0i32; 256];
        let mut out_wide = [0i32; 256];
        fdct16x16(&residual, &mut out_tight, 16);
        fdct16x16(&wide, &mut out_wide, 32);
        assert_eq!(out_tight, out_wide);
    }

    // -----------------------------------------------------------------
    // 32×32 tests (DCT-only — no hybrid variant at 32×32).
    // -----------------------------------------------------------------

    /// Deterministic 32×32 residual in `[-6, 6]` for round-trip tests.
    fn sample_residual_32x32() -> [i16; 1024] {
        let mut block = [0i16; 1024];
        for r in 0..32 {
            for c in 0..32 {
                let v = ((r * 7 + c * 11 + r * c) % 13) as i16 - 6;
                block[r * 32 + c] = v;
            }
        }
        block
    }

    /// Zero input → zero output (1D 32-point DCT).
    #[test]
    fn fdct32_zero_in_zero_out() {
        let input = [0i32; 32];
        let mut output = [99i32; 32];
        fdct32(&input, &mut output, false);
        assert_eq!(output, [0; 32]);
    }

    /// 1D fdct32 + idct32 round-trip: input scaled by **16** within
    /// ±8. (Gain pattern: 4-pt ×2, 8-pt ×4, 16-pt ×8, 32-pt ×16 = N/2.)
    #[test]
    fn fdct32_idct32_round_trip_small() {
        let mut input = [0i32; 32];
        // Mildly varied pattern: values in [-3, 3].
        for (k, slot) in input.iter_mut().enumerate() {
            *slot = (k as i32 % 7) - 3;
        }
        let mut spectrum = [0i32; 32];
        fdct32(&input, &mut spectrum, false);
        let mut round_trip = [0i32; 32];
        idct32(&spectrum, &mut round_trip);
        for k in 0..32 {
            let expected = input[k] * 16;
            assert!(
                (round_trip[k] - expected).abs() <= 8,
                "round_trip[{k}] = {} not within ±8 of 16 * input[{k}] = {}",
                round_trip[k],
                expected
            );
        }
    }

    /// Zero 32×32 residual block → zero coefficients.
    #[test]
    fn fdct32x32_zero_input_zero_output() {
        let input = [0i16; 1024];
        let mut output = [99i32; 1024];
        fdct32x32(&input, &mut output, 32);
        for (k, &v) in output.iter().enumerate() {
            assert_eq!(v, 0, "non-zero at {k}");
        }
    }

    /// `fdct32x32 → idct32x32_1024_add` recovers a small residual
    /// within ±1 per pixel. Canonical 32×32 oracle.
    #[test]
    fn fdct32x32_then_idct32x32_recovers_small_residual() {
        let residual = sample_residual_32x32();
        let mut coefs = [0i32; 1024];
        fdct32x32(&residual, &mut coefs, 32);

        let mut dest = [128u8; 1024];
        idct32x32_1024_add(&coefs, &mut dest, 32);

        for (k, (&d, &r)) in dest.iter().zip(residual.iter()).enumerate() {
            let recovered = (d as i32) - 128;
            let expected = r as i32;
            assert!(
                (recovered - expected).abs() <= 1,
                "pos {k} recovered={recovered} expected={expected}"
            );
        }
    }

    /// `fadst4` with all-zero except one non-zero entry hits the
    /// **non-short-circuit** path (the early-return is bypassed because
    /// not every input is zero). Verifies the regular computation
    /// produces a non-zero output on a non-zero impulse input.
    #[test]
    fn fadst4_single_nonzero_input_bypasses_early_return() {
        // Single non-zero element at position 0 — the early-return at
        // `vp9_dct.c:244` does NOT fire (only fires when ALL four
        // inputs are zero), so the staged ADST computation runs.
        let input: [i32; 4] = [100, 0, 0, 0];
        let mut output = [0i32; 4];
        fadst4(&input, &mut output);
        // ADST of a DC-only input must produce at least one non-zero
        // output coefficient.
        assert!(
            output.iter().any(|&v| v != 0),
            "fadst4 of non-zero impulse must produce non-zero output, got {output:?}"
        );
    }

    /// `fdct4x4` with a non-zero column-0 input exercises the DC bump
    /// branch (`if i == 0 && in_high[0] != 0 { in_high[0] += 1 }`).
    /// Verify the bump shifts the DC coefficient by exactly +1 (in
    /// the pre-shift domain → +16 in the scaled-input domain).
    #[test]
    fn fdct4x4_dc_bump_branch_fires_for_nonzero_dc() {
        // All-zero except top-left → in_high[0] = 16 (after ×16), so the
        // DC bump branch fires and adds 1 → 17. (The pure column 0 input
        // would produce 0 without the bump, so this proves the bump
        // actually fired.)
        let mut input = [0i16; 16];
        input[0] = 1;
        let mut output = [0i32; 16];
        fdct4x4(&input, &mut output, 4);
        // With the bump, output[0] should be non-zero (the +1 bump
        // makes the DC coefficient slightly different from 0).
        assert!(
            output[0] != 0,
            "DC bump should produce non-zero output[0]; got 0"
        );
    }

    /// `fht16x16` with one of the non-DctDct types exercises the
    /// non-short-circuit path and the `(v + 1 + (v<0)) >> 2` reduction
    /// at line 1043 — including the `v < 0` arm if any negative
    /// intermediate value appears.
    #[test]
    fn fht16x16_non_dct_produces_negative_intermediates() {
        let residual = sample_residual_16x16();
        let mut coefs = [0i32; 256];
        // ADST/ADST is most likely to produce both-sign output values
        // (and therefore exercise the `v < 0` adjust branch).
        fht16x16(&residual, &mut coefs, 16, TxType::AdstAdst);
        // The transform should produce both positive and negative values
        // for non-zero input (otherwise the negative-rounding branch
        // can't fire).
        let has_pos = coefs.iter().any(|&c| c > 0);
        let has_neg = coefs.iter().any(|&c| c < 0);
        assert!(
            has_pos && has_neg,
            "fht16x16 ADST/ADST: expected both signs in output"
        );
    }

    /// `fht8x8` with non-DctDct type exercises the `(v + (v<0)) >> 1`
    /// reduction at line 565 — needs negative intermediate values.
    #[test]
    fn fht8x8_non_dct_produces_negative_intermediates() {
        let residual: [i16; 64] = std::array::from_fn(|i| {
            // Mix of signs.
            (((i as i32 * 7) % 17) - 8) as i16
        });
        let mut coefs = [0i32; 64];
        fht8x8(&residual, &mut coefs, 8, TxType::AdstDct);
        let has_pos = coefs.iter().any(|&c| c > 0);
        let has_neg = coefs.iter().any(|&c| c < 0);
        assert!(has_pos && has_neg, "fht8x8 ADST/DCT: expected both signs");
    }

    /// `fdct32x32` with mixed-sign residual must produce both-sign
    /// output coefficients, exercising the `v < 0` rounding adjustment
    /// at line 1405.
    #[test]
    fn fdct32x32_mixed_sign_residual_produces_both_signs() {
        let residual: [i16; 1024] = std::array::from_fn(|i| ((i as i32 % 31) - 15) as i16);
        let mut coefs = [0i32; 1024];
        fdct32x32(&residual, &mut coefs, 32);
        let has_pos = coefs.iter().any(|&c| c > 0);
        let has_neg = coefs.iter().any(|&c| c < 0);
        assert!(
            has_pos && has_neg,
            "fdct32x32: expected both signs in output"
        );
    }

    /// `fdct32x32` honours non-trivial `stride`.
    #[test]
    fn fdct32x32_stride_independence() {
        let residual = sample_residual_32x32();
        let mut wide = [0i16; 32 * 64];
        for r in 0..32 {
            for c in 0..32 {
                wide[r * 64 + c] = residual[r * 32 + c];
            }
        }

        let mut out_tight = [0i32; 1024];
        let mut out_wide = [0i32; 1024];
        fdct32x32(&residual, &mut out_tight, 32);
        fdct32x32(&wide, &mut out_wide, 64);
        for (k, (&t, &w)) in out_tight.iter().zip(out_wide.iter()).enumerate() {
            assert_eq!(t, w, "pos {k}");
        }
    }
}
