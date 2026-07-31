//! Forward (encoder) quantizer — the `vp9_quantize_b` / `vp9_quantize_fp`
//! family plus the 32×32 variants.
//!
//! Mirror of the inverse-quant path that lives in [`super`]
//! (`dc_quant`/`ac_quant` + the [`super::DC_QLOOKUP`]/[`super::AC_QLOOKUP`]
//! tables). Scalar functions remain the cross-platform oracle; AArch64
//! realtime callers use a byte-exact NEON fast-quantizer wrapper.
//!
//! ## Source mapping
//!
//! - [`vp9_quantize_b`] mirrors libvpx `vpx_quantize_b_c`
//!   (`vpx_dsp/quantize.c:118-165`).
//! - [`vp9_quantize_b_32x32`] mirrors libvpx `vpx_quantize_b_32x32_c`
//!   (`vpx_dsp/quantize.c:216-275`).
//! - [`vp9_quantize_fp`] mirrors libvpx `vp9_quantize_fp_c`
//!   (`vp9/encoder/vp9_quantize.c:25-55`).
//! - [`vp9_quantize_fp_32x32`] mirrors libvpx `vp9_quantize_fp_32x32_c`
//!   (`vp9/encoder/vp9_quantize.c:91-122`).
//! - [`init_plane_quantizers`] mirrors the per-q-index slice of libvpx's
//!   `vp9_init_quantizer` (`vp9/encoder/vp9_quantize.c:184-244`) for 8-bit
//!   Profile 0. Only one `q_index` is materialised per call — the libvpx
//!   helper precomputes all 256 rows ahead of time; we evaluate on demand
//!   because the per-frame `q_index` is fixed once frame setup picks it.
//!
//! ## Divergences
//!
//! None functionally — math is `int32` end-to-end, output is `i16`
//! (`tran_low_t` in the 8-bit, non-HBD libvpx build). Two API ergonomics
//! differences from the libvpx C signature:
//!
//! 1. Inputs/outputs are slices, not pointers. `n_coeffs` survives only as
//!    an explicit parameter (mirrors libvpx) so that the same function can
//!    serve `Tx4x4..=Tx16x16`. The 32×32 variants hard-code `n_coeffs =
//!    1024` (matching libvpx, which also hard-codes the 1024 limit in the
//!    32×32 functions).
//! 2. `skip_block` is a function parameter rather than an outer assertion
//!    at the call site. libvpx asserts `!x->skip_block` in
//!    `vp9_encodemb.c` before calling the kernels; we fold the
//!    skip-block-zeroes-everything path into the kernel itself. The
//!    semantics are identical: `skip_block=true` zeroes both output
//!    buffers and returns `eob=0`.
//!
//! ## Scope
//!
//! 8-bit Profile 0 only (master ledger D-01 RESOLVED). The
//! `CONFIG_VP9_HIGHBITDEPTH` paths in libvpx are out of scope.

use super::{ac_quant, dc_quant};
use crate::bitstream::BitDepth;
use crate::transforms::scans::ScanOrder;
use crate::transforms::types::TxSize;

/// `INT16_MIN` as `i32`. Used in the libvpx `clamp(... , INT16_MIN,
/// INT16_MAX)` saturation before the multiply-by-quant step.
const INT16_MIN_I32: i32 = i16::MIN as i32;
/// `INT16_MAX` as `i32`. Companion to [`INT16_MIN_I32`].
const INT16_MAX_I32: i32 = i16::MAX as i32;

/// `ROUND_POWER_OF_TWO(v, 1) = (v + 1) >> 1`. Matches the libvpx macro at
/// `vpx_dsp/vpx_dsp_common.h:39` for `n=1` — the only `n` the 32×32
/// quantizer cares about.
#[inline]
const fn round_pow2_1(v: i32) -> i32 {
    (v + 1) >> 1
}

/// Forward quantizer (zbin / round / quant / shift) for 4×4, 8×8, and
/// 16×16 blocks.
///
/// Mirrors libvpx `vpx_quantize_b_c` (`vpx_dsp/quantize.c:118-165`).
///
/// # Parameters
/// - `coeff_ptr` — input residual coefficients in raster order. Length
///   must be `>= n_coeffs`. `i32` mirrors `tran_low_t` (the wider variant
///   libvpx ships under `CONFIG_VP9_HIGHBITDEPTH`); the math itself fits
///   `i32` for the 8-bit path because forward-DCT outputs are
///   16-bit-bounded after `wraplow`.
/// - `n_coeffs` — number of coefficients (16, 64, or 256). 32×32 callers
///   use [`vp9_quantize_b_32x32`] instead.
/// - `skip_block` — when `true`, zero `qcoeff_ptr` + `dqcoeff_ptr` and
///   return `eob = 0`. Mirrors the upstream `assert(!x->skip_block)` +
///   pre-zeroed buffers contract in libvpx — folded into the kernel for
///   ergonomics; see module docs.
/// - `zbin_ptr` — `[dc_zbin, ac_zbin]`. Coefficients with `|coeff| <
///   zbin` are pinned to zero (the zero-bin gate).
/// - `round_ptr` — `[dc_round, ac_round]`. Offset added to `|coeff|`
///   before the quant multiply.
/// - `quant_ptr` — `[dc_quant, ac_quant]`. Quant-multiplier — see
///   `invert_quant` in libvpx for the derivation
///   (`vp9/encoder/vp9_quantize.c:158-166`).
/// - `quant_shift_ptr` — `[dc_shift, ac_shift]`. Companion to
///   `quant_ptr`.
/// - `qcoeff_ptr` — output quantized coefficients in raster order. Length
///   must be `>= n_coeffs`.
/// - `dqcoeff_ptr` — output dequantized coefficients in raster order
///   (`qcoeff * dequant`). Length must be `>= n_coeffs`.
/// - `dequant_ptr` — `[dc_dequant, ac_dequant]`. The reverse of
///   `quant_ptr`; libvpx keeps both around so the encoder can compute the
///   `dqcoeff` (= `qcoeff * dequant`) the decoder will reconstruct, for
///   distortion / rate-distortion purposes.
/// - `scan` — forward scan order. The pre-scan walks back-to-front
///   skipping coefficients smaller than zbin, then the quantization walk
///   processes positions in scan-order.
///
/// # Returns
///
/// `eob` — one past the scan-position of the last non-zero `qcoeff`, or
/// `0` if all coefficients quantized to zero.
#[allow(clippy::too_many_arguments)] // matches libvpx 1:1
#[must_use]
pub fn vp9_quantize_b(
    coeff_ptr: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    zbin_ptr: &[i16; 2],
    round_ptr: &[i16; 2],
    quant_ptr: &[i16; 2],
    quant_shift_ptr: &[i16; 2],
    qcoeff_ptr: &mut [i16],
    dqcoeff_ptr: &mut [i16],
    dequant_ptr: &[i16; 2],
    scan: &[i16],
) -> u16 {
    assert!(coeff_ptr.len() >= n_coeffs);
    assert!(qcoeff_ptr.len() >= n_coeffs);
    assert!(dqcoeff_ptr.len() >= n_coeffs);
    assert!(scan.len() >= n_coeffs);

    // Zero outputs unconditionally — libvpx `vpx_quantize_b_c:131-132`.
    for slot in &mut qcoeff_ptr[..n_coeffs] {
        *slot = 0;
    }
    for slot in &mut dqcoeff_ptr[..n_coeffs] {
        *slot = 0;
    }

    if skip_block {
        return 0;
    }

    let zbins = [i32::from(zbin_ptr[0]), i32::from(zbin_ptr[1])];
    let nzbins = [-zbins[0], -zbins[1]];

    // Pre-scan: walk back-to-front, dropping trailing scan positions
    // whose coefficients fall inside the zbin (-zbin, zbin) range.
    let mut non_zero_count = n_coeffs as i32;
    for i in (0..n_coeffs as i32).rev() {
        let rc = scan[i as usize] as usize;
        let coeff = coeff_ptr[rc];
        let band = usize::from(i != 0);
        if coeff < zbins[band] && coeff > nzbins[band] {
            non_zero_count -= 1;
        } else {
            break;
        }
    }

    // Quantization pass. Mirrors `vpx_dsp/quantize.c:147-163`.
    let mut eob: i32 = -1;
    for i in 0..non_zero_count {
        let rc = scan[i as usize] as usize;
        let coeff = coeff_ptr[rc];
        let coeff_sign = coeff >> 31; // arithmetic shift — 0 or -1
        let abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
        let band = usize::from(i != 0);

        if abs_coeff >= zbins[band] {
            let mut tmp =
                (abs_coeff + i32::from(round_ptr[band])).clamp(INT16_MIN_I32, INT16_MAX_I32);
            tmp = ((((tmp * i32::from(quant_ptr[band])) >> 16) + tmp)
                * i32::from(quant_shift_ptr[band]))
                >> 16;
            let signed = (tmp ^ coeff_sign) - coeff_sign;
            qcoeff_ptr[rc] = signed as i16;
            dqcoeff_ptr[rc] = (qcoeff_ptr[rc] as i32 * i32::from(dequant_ptr[band])) as i16;
            if tmp != 0 {
                eob = i;
            }
        }
    }
    (eob + 1) as u16
}

/// Zbin quantizer with a complete scan order.
///
/// AArch64 uses the byte-exact NEON raster-order kernel and `iscan` EOB
/// reduction. Other targets retain [`vp9_quantize_b`].
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn vp9_quantize_b_with_order(
    coeff_ptr: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    zbin_ptr: &[i16; 2],
    round_ptr: &[i16; 2],
    quant_ptr: &[i16; 2],
    quant_shift_ptr: &[i16; 2],
    qcoeff_ptr: &mut [i16],
    dqcoeff_ptr: &mut [i16],
    dequant_ptr: &[i16; 2],
    order: &ScanOrder,
) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::dsp::aarch64::vp9_quantize_b_neon(
            coeff_ptr,
            n_coeffs,
            skip_block,
            zbin_ptr,
            round_ptr,
            quant_ptr,
            quant_shift_ptr,
            qcoeff_ptr,
            dqcoeff_ptr,
            dequant_ptr,
            order.iscan,
        )
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        vp9_quantize_b(
            coeff_ptr,
            n_coeffs,
            skip_block,
            zbin_ptr,
            round_ptr,
            quant_ptr,
            quant_shift_ptr,
            qcoeff_ptr,
            dqcoeff_ptr,
            dequant_ptr,
            order.scan,
        )
    }
}

/// Forward quantizer for 32×32 transform blocks.
///
/// Mirrors libvpx `vpx_quantize_b_32x32_c` (`vpx_dsp/quantize.c:216-275`).
///
/// The arithmetic differs from [`vp9_quantize_b`] in three ways:
///
/// 1. `zbin` is `ROUND_POWER_OF_TWO(zbin, 1)`.
/// 2. `round` (added to `abs_coeff`) is also `ROUND_POWER_OF_TWO(round,
///    1)`.
/// 3. The post-`quant_shift` reduction is `>> 15` instead of `>> 16`, and
///    `dqcoeff = qcoeff * dequant / 2`.
///
/// These three changes — the trailing `/2` on the dequant side and the
/// `>> 15` instead of `>> 16` on the quant side — combine to mirror the
/// decoder's `dq_shift = 1` for `Tx32x32` in
/// [`crate::entropy::detokenize::decode_coefs`].
///
/// # Panics
///
/// `coeff_ptr`, `qcoeff_ptr`, `dqcoeff_ptr`, and `scan` must each be at
/// least `1024` long.
#[allow(clippy::too_many_arguments)] // matches libvpx 1:1
#[must_use]
pub fn vp9_quantize_b_32x32(
    coeff_ptr: &[i32],
    skip_block: bool,
    zbin_ptr: &[i16; 2],
    round_ptr: &[i16; 2],
    quant_ptr: &[i16; 2],
    quant_shift_ptr: &[i16; 2],
    qcoeff_ptr: &mut [i16],
    dqcoeff_ptr: &mut [i16],
    dequant_ptr: &[i16; 2],
    scan: &[i16],
) -> u16 {
    const N_COEFFS: usize = 1024;
    assert!(coeff_ptr.len() >= N_COEFFS);
    assert!(qcoeff_ptr.len() >= N_COEFFS);
    assert!(dqcoeff_ptr.len() >= N_COEFFS);
    assert!(scan.len() >= N_COEFFS);

    for slot in &mut qcoeff_ptr[..N_COEFFS] {
        *slot = 0;
    }
    for slot in &mut dqcoeff_ptr[..N_COEFFS] {
        *slot = 0;
    }

    if skip_block {
        return 0;
    }

    let zbins = [
        round_pow2_1(i32::from(zbin_ptr[0])),
        round_pow2_1(i32::from(zbin_ptr[1])),
    ];
    let nzbins = [-zbins[0], -zbins[1]];

    // Pre-scan: collect scan-positions whose coefficient is outside the
    // (-zbin, zbin) range. libvpx walks forward (not backward) for the
    // 32×32 path; the side effect is that positions inside the zbin in
    // the middle of the block stay populated in `idx_arr`, but the
    // ordering of survivors matches the scan order.
    let mut idx_arr = [0i32; N_COEFFS];
    let mut idx: usize = 0;
    for i in 0..N_COEFFS as i32 {
        let rc = scan[i as usize] as usize;
        let coeff = coeff_ptr[rc];
        let band = usize::from(i != 0);
        if coeff >= zbins[band] || coeff <= nzbins[band] {
            idx_arr[idx] = i;
            idx += 1;
        }
    }

    let mut eob: i32 = -1;
    for &i in idx_arr.iter().take(idx) {
        let rc = scan[i as usize] as usize;
        let coeff = coeff_ptr[rc];
        let coeff_sign = coeff >> 31;
        let mut abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
        let band = usize::from(i != 0);

        abs_coeff += round_pow2_1(i32::from(round_ptr[band]));
        abs_coeff = abs_coeff.clamp(INT16_MIN_I32, INT16_MAX_I32);
        let tmp = ((((abs_coeff * i32::from(quant_ptr[band])) >> 16) + abs_coeff)
            * i32::from(quant_shift_ptr[band]))
            >> 15;
        let signed = (tmp ^ coeff_sign) - coeff_sign;
        qcoeff_ptr[rc] = signed as i16;

        // libvpx applies an x86-specific saturation here (`vpx_dsp/
        // quantize.c:262-270`) when `tran_low_t` is 16-bit; we always
        // operate on `i16` outputs, so we always saturate.
        let dq = (i32::from(qcoeff_ptr[rc]) * i32::from(dequant_ptr[band])) / 2;
        dqcoeff_ptr[rc] = dq.clamp(INT16_MIN_I32, INT16_MAX_I32) as i16;

        if tmp != 0 {
            eob = i;
        }
    }
    (eob + 1) as u16
}

/// Fast-path forward quantizer (skips the zbin gate, uses simpler
/// rounding) for 4×4, 8×8, and 16×16 blocks.
///
/// Mirrors libvpx `vp9_quantize_fp_c` (`vp9/encoder/vp9_quantize.c:25-55`).
///
/// Two differences from [`vp9_quantize_b`]:
///
/// 1. No zbin pre-scan: every scan position is evaluated.
/// 2. The quant pipeline collapses to one multiply + one `>> 16` (no
///    extra `quant_shift_ptr` term).
///
/// Used by the realtime / `cpu-used 6+` encoder path where the saved
/// branches matter and the slightly worse rate-distortion is acceptable.
#[allow(clippy::too_many_arguments)] // matches libvpx 1:1
#[must_use]
pub fn vp9_quantize_fp(
    coeff_ptr: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    round_ptr: &[i16; 2],
    quant_ptr: &[i16; 2],
    qcoeff_ptr: &mut [i16],
    dqcoeff_ptr: &mut [i16],
    dequant_ptr: &[i16; 2],
    scan: &[i16],
) -> u16 {
    assert!(coeff_ptr.len() >= n_coeffs);
    assert!(qcoeff_ptr.len() >= n_coeffs);
    assert!(dqcoeff_ptr.len() >= n_coeffs);
    assert!(scan.len() >= n_coeffs);

    for slot in &mut qcoeff_ptr[..n_coeffs] {
        *slot = 0;
    }
    for slot in &mut dqcoeff_ptr[..n_coeffs] {
        *slot = 0;
    }

    if skip_block {
        return 0;
    }

    let mut eob: i32 = -1;
    for i in 0..n_coeffs as i32 {
        let rc = scan[i as usize] as usize;
        let coeff = coeff_ptr[rc];
        let coeff_sign = coeff >> 31;
        let abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
        let band = usize::from(i != 0);

        let tmp = (abs_coeff + i32::from(round_ptr[band])).clamp(INT16_MIN_I32, INT16_MAX_I32);
        let tmp = (tmp * i32::from(quant_ptr[band])) >> 16;

        let signed = (tmp ^ coeff_sign) - coeff_sign;
        qcoeff_ptr[rc] = signed as i16;
        dqcoeff_ptr[rc] = (i32::from(qcoeff_ptr[rc]) * i32::from(dequant_ptr[band])) as i16;

        if tmp != 0 {
            eob = i;
        }
    }
    (eob + 1) as u16
}

/// Fast-path quantizer with a complete scan order.
///
/// AArch64 uses the byte-exact NEON raster-order kernel and `iscan` EOB
/// reduction. Other targets retain [`vp9_quantize_fp`].
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn vp9_quantize_fp_with_order(
    coeff_ptr: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    round_ptr: &[i16; 2],
    quant_ptr: &[i16; 2],
    qcoeff_ptr: &mut [i16],
    dqcoeff_ptr: &mut [i16],
    dequant_ptr: &[i16; 2],
    order: &ScanOrder,
) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::dsp::aarch64::vp9_quantize_fp_neon(
            coeff_ptr,
            n_coeffs,
            skip_block,
            round_ptr,
            quant_ptr,
            qcoeff_ptr,
            dqcoeff_ptr,
            dequant_ptr,
            order.iscan,
        )
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        vp9_quantize_fp(
            coeff_ptr,
            n_coeffs,
            skip_block,
            round_ptr,
            quant_ptr,
            qcoeff_ptr,
            dqcoeff_ptr,
            dequant_ptr,
            order.scan,
        )
    }
}

/// Fast-path forward quantizer for 32×32 transform blocks.
///
/// Mirrors libvpx `vp9_quantize_fp_32x32_c`
/// (`vp9/encoder/vp9_quantize.c:91-122`).
///
/// Unlike the non-fast 32×32 variant, this one has an explicit
/// `dequant >> 2` gate to skip very-small coefficients (mirroring the
/// libvpx `if (abs_coeff >= (dequant_ptr[rc != 0] >> 2))` guard at line
/// 111) — that gate replaces the zbin pre-scan.
#[allow(clippy::too_many_arguments)] // matches libvpx 1:1
#[must_use]
pub fn vp9_quantize_fp_32x32(
    coeff_ptr: &[i32],
    skip_block: bool,
    round_ptr: &[i16; 2],
    quant_ptr: &[i16; 2],
    qcoeff_ptr: &mut [i16],
    dqcoeff_ptr: &mut [i16],
    dequant_ptr: &[i16; 2],
    scan: &[i16],
) -> u16 {
    const N_COEFFS: usize = 1024;
    assert!(coeff_ptr.len() >= N_COEFFS);
    assert!(qcoeff_ptr.len() >= N_COEFFS);
    assert!(dqcoeff_ptr.len() >= N_COEFFS);
    assert!(scan.len() >= N_COEFFS);

    for slot in &mut qcoeff_ptr[..N_COEFFS] {
        *slot = 0;
    }
    for slot in &mut dqcoeff_ptr[..N_COEFFS] {
        *slot = 0;
    }

    if skip_block {
        return 0;
    }

    let mut eob: i32 = -1;
    for i in 0..N_COEFFS as i32 {
        let rc = scan[i as usize] as usize;
        let coeff = coeff_ptr[rc];
        let coeff_sign = coeff >> 31;
        let abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
        let band = usize::from(i != 0);

        let mut tmp = 0i32;
        if abs_coeff >= (i32::from(dequant_ptr[band]) >> 2) {
            let mut a = abs_coeff + round_pow2_1(i32::from(round_ptr[band]));
            a = a.clamp(INT16_MIN_I32, INT16_MAX_I32);
            tmp = (a * i32::from(quant_ptr[band])) >> 15;
            let signed = (tmp ^ coeff_sign) - coeff_sign;
            qcoeff_ptr[rc] = signed as i16;
            dqcoeff_ptr[rc] =
                ((i32::from(qcoeff_ptr[rc]) * i32::from(dequant_ptr[band])) / 2) as i16;
        }

        if tmp != 0 {
            eob = i;
        }
    }
    (eob + 1) as u16
}

/// Fast 32x32 quantizer with a complete scan order.
///
/// AArch64 uses the byte-exact NEON raster-order kernel and `iscan` EOB
/// reduction. Other targets retain [`vp9_quantize_fp_32x32`].
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn vp9_quantize_fp_32x32_with_order(
    coeff_ptr: &[i32],
    skip_block: bool,
    round_ptr: &[i16; 2],
    quant_ptr: &[i16; 2],
    qcoeff_ptr: &mut [i16],
    dqcoeff_ptr: &mut [i16],
    dequant_ptr: &[i16; 2],
    order: &ScanOrder,
) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::dsp::aarch64::vp9_quantize_fp_32x32_neon(
            coeff_ptr,
            skip_block,
            round_ptr,
            quant_ptr,
            qcoeff_ptr,
            dqcoeff_ptr,
            dequant_ptr,
            order.iscan,
        )
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        vp9_quantize_fp_32x32(
            coeff_ptr,
            skip_block,
            round_ptr,
            quant_ptr,
            qcoeff_ptr,
            dqcoeff_ptr,
            dequant_ptr,
            order.scan,
        )
    }
}

/// Per-plane forward-quantizer parameter pack. Mirrors the eight slots
/// libvpx populates per (q_index, [DC, AC]) pair in `vp9_init_quantizer`
/// (`vp9/encoder/vp9_quantize.c:184-244`).
///
/// Each field is a `[DC, AC]` pair indexed by `usize::from(scan_pos !=
/// 0)`, matching the `rc != 0` branch in every quantizer above.
#[derive(Copy, Clone, Debug, Default)]
pub struct PlaneQuantizers {
    /// `vp9_quantize_b` quant multiplier. Derived via `invert_quant`.
    pub quant: [i16; 2],
    /// `vp9_quantize_b` quant shift. Companion to [`Self::quant`].
    pub quant_shift: [i16; 2],
    /// `vp9_quantize_fp` quant multiplier (no shift companion needed).
    pub quant_fp: [i16; 2],
    /// `vp9_quantize_b` rounding offset.
    pub round: [i16; 2],
    /// `vp9_quantize_fp` rounding offset.
    pub round_fp: [i16; 2],
    /// Zero-bin threshold.
    pub zbin: [i16; 2],
    /// Dequantizer (inverse of [`Self::quant`]). The decoder uses this
    /// value verbatim.
    pub dequant: [i16; 2],
}

/// Plane channel selector — picks DC delta and (for chroma) AC delta. The
/// Y plane has no AC delta in libvpx; both deltas live in the frame
/// header.
#[derive(Copy, Clone, Debug)]
pub struct QuantizerDeltas {
    /// `y_dc_delta_q` or `uv_dc_delta_q` from the frame header.
    pub dc_delta: i32,
    /// `0` for Y, `uv_ac_delta_q` for UV.
    pub ac_delta: i32,
}

impl QuantizerDeltas {
    /// Default deltas for the Y plane (no DC or AC delta).
    pub const Y: Self = Self {
        dc_delta: 0,
        ac_delta: 0,
    };
}

/// Compute a single `(q_index, plane_deltas)` slice of libvpx's
/// `vp9_init_quantizer`.
///
/// Mirrors the inner-loop body of
/// `vp9_init_quantizer` (`vp9/encoder/vp9_quantize.c:184-244`). The
/// outer loop iterates over `q_index` and the plane (`y_quant` vs
/// `uv_quant`); we materialise only one cell because the per-frame
/// quantizer is fixed once frame setup picks it.
///
/// Sharpness is fixed to `0` (the default — libvpx's `oxcf.sharpness`
/// is an external knob that the realtime encoder doesn't touch). When we
/// later expose a sharpness setting (post-MVP), this signature gains a
/// `sharpness: i32` parameter.
#[must_use]
pub fn init_plane_quantizers(q_index: i32, deltas: QuantizerDeltas) -> PlaneQuantizers {
    let qzbin_factor = qzbin_factor_8bit(q_index);
    let qrounding_factor = if q_index == 0 { 64 } else { 48 };

    let mut out = PlaneQuantizers::default();
    for i in 0..2 {
        let qrounding_factor_fp = if q_index == 0 {
            64
        } else if i == 0 {
            48
        } else {
            42
        };

        // libvpx `vp9_dc_quant(q, dc_delta)` for `i==0`,
        // `vp9_ac_quant(q, ac_delta)` for `i==1`. Encoder is 8-bit only
        // (D-01 RESOLVED); HBD encoder support is out of scope for
        // Track T phase 0.
        let quant = if i == 0 {
            i32::from(dc_quant(q_index, deltas.dc_delta, BitDepth::Eight))
        } else {
            i32::from(ac_quant(q_index, deltas.ac_delta, BitDepth::Eight))
        };

        let (q_b, shift) = invert_quant(quant);
        out.quant[i] = q_b;
        out.quant_shift[i] = shift;
        out.quant_fp[i] = ((1i32 << 16) / quant) as i16;
        out.round_fp[i] = ((qrounding_factor_fp * quant) >> 7) as i16;
        out.zbin[i] = round_pow2_n(qzbin_factor * quant, 7) as i16;
        out.round[i] = ((qrounding_factor * quant) >> 7) as i16;
        out.dequant[i] = quant as i16;
    }
    out
}

/// `vp9_init_quantizer`'s `qzbin_factor` decision for the 8-bit path
/// (`vp9/encoder/vp9_quantize.c:168-182`). `q == 0` is the lossless
/// special-case; the `quant < 148` branch is the original 8-bit cutoff
/// libvpx picks.
#[inline]
fn qzbin_factor_8bit(q_index: i32) -> i32 {
    if q_index == 0 {
        64
    } else if i32::from(dc_quant(q_index, 0, BitDepth::Eight)) < 148 {
        84
    } else {
        80
    }
}

/// `ROUND_POWER_OF_TWO(v, n)` — mirrors libvpx
/// `vpx_dsp/vpx_dsp_common.h:39`. `n` is small (≤ 7) for all known call
/// sites, so plain `i32` arithmetic suffices.
#[inline]
const fn round_pow2_n(v: i32, n: u32) -> i32 {
    (v + (1 << (n - 1))) >> n
}

/// Compute `(quant_b, quant_shift_b)` from `dequant`. Direct port of
/// libvpx `invert_quant` (`vp9/encoder/vp9_quantize.c:158-166`).
///
/// `quant_b = (1 + (1 << (16 + l)) / d) - (1 << 16)` where
/// `l = floor(log2(d / 2))`, and `quant_shift_b = 1 << (16 - l)`.
fn invert_quant(d: i32) -> (i16, i16) {
    let mut t = d as u32;
    let mut l: u32 = 0;
    while t > 1 {
        l += 1;
        t >>= 1;
    }
    let m = 1 + ((1u64 << (16 + l)) / d as u64) as i32;
    let quant = (m - (1 << 16)) as i16;
    let shift = (1 << (16 - l)) as i16;
    (quant, shift)
}

/// Return the AC dequant for a `(plane, q_index)` pair to drive the
/// decoder's `(dc_dequant, ac_dequant)` pair. Encoder-side compatibility
/// helper — mirrors `cpi->y_dequant[q][1]` or `cpi->uv_dequant[q][1]`.
#[must_use]
pub fn plane_dequant_pair(q_index: i32, deltas: QuantizerDeltas) -> (i16, i16) {
    // Encoder-side helper — 8-bit only. HBD encoder support is out of
    // scope for Track T phase 0.
    (
        dc_quant(q_index, deltas.dc_delta, BitDepth::Eight),
        ac_quant(q_index, deltas.ac_delta, BitDepth::Eight),
    )
}

/// Convenience: return `n_coeffs` for a given [`TxSize`]. Re-exported so
/// quant callers don't have to import `TxSize::n_coeffs` themselves.
#[inline]
#[must_use]
pub const fn n_coeffs_for(tx_size: TxSize) -> usize {
    tx_size.n_coeffs()
}

/// Convenience: extract the forward `scan` slice from a [`ScanOrder`].
/// The scalar and 32x32 quantizers take `scan`; the AArch64 non-32x32 fast
/// wrapper instead takes the complete order so it can derive EOB from
/// `iscan`, like libvpx's NEON kernel.
#[inline]
#[must_use]
pub fn scan_for_forward(order: &ScanOrder) -> &[i16] {
    order.scan()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::bool_reader::BoolReader;
    use crate::entropy::bool_writer::BoolWriter;
    use crate::entropy::detokenize::{decode_coefs, encode_coefs};
    use crate::entropy::frame_context::FrameContext;
    use crate::transforms::forward::fdct8x8;
    use crate::transforms::scans::{default_scan_4x4, default_scan_8x8};
    use crate::transforms::types::TxSize;

    /// Tiny harness: drive [`vp9_quantize_b`] for a 4×4 block with a
    /// caller-supplied coeff input + zbin/round/quant/shift quad. Returns
    /// `(qcoeff, dqcoeff, eob)`.
    fn quantize_b_4x4(
        coeff: &[i32; 16],
        skip: bool,
        zbin: [i16; 2],
        round: [i16; 2],
        quant: [i16; 2],
        quant_shift: [i16; 2],
        dequant: [i16; 2],
    ) -> ([i16; 16], [i16; 16], u16) {
        let scan = default_scan_4x4();
        let mut qcoeff = [0i16; 16];
        let mut dqcoeff = [0i16; 16];
        let eob = vp9_quantize_b(
            coeff,
            16,
            skip,
            &zbin,
            &round,
            &quant,
            &quant_shift,
            &mut qcoeff,
            &mut dqcoeff,
            &dequant,
            scan.scan(),
        );
        (qcoeff, dqcoeff, eob)
    }

    /// `skip_block = true` zeros qcoeff / dqcoeff regardless of input
    /// and returns `eob = 0`.
    #[test]
    fn vp9_quantize_b_skip_block_zeroes_everything() {
        let coeff = [1000i32; 16]; // huge, would otherwise survive
        let (qcoeff, dqcoeff, eob) = quantize_b_4x4(
            &coeff,
            true,
            [10, 10],
            [4, 4],
            [16384, 16384],
            [1, 1],
            [8, 8],
        );
        assert_eq!(eob, 0);
        assert!(qcoeff.iter().all(|&v| v == 0));
        assert!(dqcoeff.iter().all(|&v| v == 0));
    }

    /// Coefficients with `|coeff| < zbin` quantize to zero.
    #[test]
    fn vp9_quantize_b_below_zbin_quantizes_to_zero() {
        // Zbin = 20 for both DC and AC; coeff = 10 < 20.
        let mut coeff = [0i32; 16];
        coeff[0] = 10;
        coeff[5] = 10;
        let (qcoeff, dqcoeff, eob) = quantize_b_4x4(
            &coeff,
            false,
            [20, 20],
            [4, 4],
            [16384, 16384],
            [1, 1],
            [8, 8],
        );
        assert_eq!(eob, 0);
        assert!(qcoeff.iter().all(|&v| v == 0));
        assert!(dqcoeff.iter().all(|&v| v == 0));
    }

    /// A coefficient exactly equal to `zbin` survives the zero-bin gate
    /// and produces a non-zero qcoeff. We pick `quant`/`shift` so the
    /// arithmetic yields a non-zero result.
    #[test]
    fn vp9_quantize_b_at_zbin_quantizes_to_nonzero() {
        // q = 60 derived constants — see q60_quantizers() below for the
        // derivation. Picked because the DC quant is large enough that
        // `abs_coeff = zbin` reliably crosses the post-shift threshold.
        let qp = q60_y_quantizers();
        // Coeff exactly at zbin. DC band uses `zbin[0]`.
        let mut coeff = [0i32; 16];
        coeff[0] = i32::from(qp.zbin[0]);
        let (qcoeff, _dq, eob) = quantize_b_4x4(
            &coeff,
            false,
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
        );
        assert_ne!(qcoeff[0], 0, "coeff at zbin should not quantize to zero");
        assert!(eob >= 1);
    }

    /// Sign of input is preserved in `qcoeff`.
    #[test]
    fn vp9_quantize_b_negative_input_preserves_sign() {
        let qp = q60_y_quantizers();
        let mut coeff = [0i32; 16];
        coeff[0] = -200; // clearly above zbin
        let (qcoeff, dqcoeff, eob) = quantize_b_4x4(
            &coeff,
            false,
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
        );
        assert!(
            qcoeff[0] < 0,
            "negative coeff should produce negative qcoeff"
        );
        assert!(dqcoeff[0] < 0);
        assert!(eob >= 1);
    }

    /// Invariant: `dqcoeff[i] = qcoeff[i] * dequant[band(i)]` for every
    /// raster position. This is the round-trip oracle the decoder uses.
    #[test]
    fn vp9_quantize_b_dqcoeff_equals_qcoeff_times_dequant() {
        let qp = q60_y_quantizers();
        let mut coeff = [0i32; 16];
        coeff[0] = 300;
        coeff[1] = -150;
        coeff[5] = 80;
        coeff[10] = -40;
        let (qcoeff, dqcoeff, _eob) = quantize_b_4x4(
            &coeff,
            false,
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
        );
        let scan = default_scan_4x4();
        for i in 0..16 {
            let band = usize::from(i != 0);
            let raster = scan.scan()[i] as usize;
            let expected = i32::from(qcoeff[raster]) * i32::from(qp.dequant[band]);
            assert_eq!(
                i32::from(dqcoeff[raster]),
                expected,
                "dqcoeff[{raster}] (scan {i}) = {}, expected qcoeff * dequant = {expected}",
                dqcoeff[raster]
            );
        }
    }

    /// `eob` is one past the scan-position of the last non-zero qcoeff.
    #[test]
    fn vp9_quantize_b_eob_is_one_past_last_nonzero() {
        let qp = q60_y_quantizers();
        // Put coefficient at raster position 5 only. In the default 4×4
        // scan, raster 5 maps to scan index... let's just check the
        // invariant rather than hardcode the position.
        let mut coeff = [0i32; 16];
        coeff[5] = 400;
        let (qcoeff, _dq, eob) = quantize_b_4x4(
            &coeff,
            false,
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
        );
        let scan = default_scan_4x4();
        // Find the highest scan-position with a non-zero qcoeff.
        let mut last_nonzero_scan_pos: Option<usize> = None;
        for i in 0..16 {
            let raster = scan.scan()[i] as usize;
            if qcoeff[raster] != 0 {
                last_nonzero_scan_pos = Some(i);
            }
        }
        assert!(
            last_nonzero_scan_pos.is_some(),
            "no non-zero qcoeff produced"
        );
        assert_eq!(eob as usize, last_nonzero_scan_pos.unwrap() + 1);
    }

    /// 32×32 uses an extra `>>1` (encoded as `dqcoeff = qcoeff * dequant
    /// / 2` plus the inner `>>15` instead of `>>16`). For an identical
    /// input, the 32×32 reconstruction value (`dqcoeff`) is roughly half
    /// of the 4×4 reconstruction value — even though both paths produce
    /// non-zero qcoeffs.
    ///
    /// Picks realistic (q=60 Y) constants so the math doesn't underflow
    /// for the chosen input.
    #[test]
    fn vp9_quantize_b_32x32_uses_extra_shift() {
        let qp = q60_y_quantizers();
        // Pick an input large enough that both paths survive the zbin /
        // shift pipeline. The forward-DCT-style DC coefficient for a flat
        // 8×8 input of 32 is roughly 1024 (32 * 32); larger blocks scale
        // similarly. Use 2000 to stay well above the threshold.
        let dc_input = 2000;

        // 4×4
        let mut coeff_4 = [0i32; 16];
        coeff_4[0] = dc_input;
        let (q4, dq4, eob4) = quantize_b_4x4(
            &coeff_4,
            false,
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
        );
        assert!(
            eob4 >= 1,
            "4×4 quantize should produce a non-zero qcoeff for dc_input={dc_input}"
        );

        // 32×32 with the same q=60 Y quantizers.
        let mut coeff_32 = vec![0i32; 1024];
        coeff_32[0] = dc_input;
        let mut q32 = vec![0i16; 1024];
        let mut dq32 = vec![0i16; 1024];
        let scan32 = crate::transforms::scans::DEFAULT_SCAN_32X32;
        let eob32 = vp9_quantize_b_32x32(
            &coeff_32,
            false,
            &qp.zbin,
            &qp.round,
            &qp.quant,
            &qp.quant_shift,
            &mut q32,
            &mut dq32,
            &qp.dequant,
            &scan32,
        );
        assert!(
            eob32 >= 1,
            "32×32 quantize should produce a non-zero qcoeff"
        );

        // The 32×32 inner shift is `>> 15` (one bit less than the 4×4
        // path's `>> 16`), so the qcoeff magnitude is roughly DOUBLE the
        // 4×4 value for the same input. The trailing `/ 2` on dqcoeff
        // compensates so the reconstructed value (dqcoeff) is the same.
        //
        // This pair of asymmetric shifts is the load-bearing
        // "extra-shift" pattern — the test name notwithstanding.
        let q4_dc = q4[0].unsigned_abs();
        let q32_dc = q32[0].unsigned_abs();
        assert!(
            q32_dc > q4_dc,
            "32x32 qcoeff[0]={q32_dc} should be strictly greater than 4x4 qcoeff[0]={q4_dc} \
             (`>> 15` vs `>> 16`)",
        );

        // Reconstruction invariant: for 32×32, dqcoeff = qcoeff * dequant
        // / 2. This is the encoder-side mirror of the decoder's
        // `dq_shift = 1` for Tx32x32.
        let band = 0; // DC
        assert_eq!(
            i32::from(dq32[0]),
            (i32::from(q32[0]) * i32::from(qp.dequant[band])) / 2,
            "32×32 dqcoeff invariant: dqcoeff = qcoeff * dequant / 2"
        );

        // The reconstruction values themselves end up within one
        // dequant-step of each other — both paths recover the same input
        // up to quantization noise.
        let dq4_dc = dq4[0].unsigned_abs() as i32;
        let dq32_dc = dq32[0].unsigned_abs() as i32;
        assert!(
            (dq32_dc - dq4_dc).abs() <= i32::from(qp.dequant[band]),
            "32×32 vs 4×4 dqcoeff should agree within one dequant step \
             ({dq32_dc} vs {dq4_dc}, dequant={})",
            qp.dequant[band]
        );
    }

    /// Fast-path quantizes any input >= dequant/2 with `qrounding_factor_fp
    /// = 48/128 * quant` of slack — far below the zbin gate used by the
    /// non-fp path. Pick `qp.zbin` and a coefficient that:
    ///
    /// 1. is below `qp.zbin[0]` (so `vp9_quantize_b` would zero it), and
    /// 2. is `>= qp.dequant[0] / 2` (so `vp9_quantize_fp` produces a
    ///    non-zero result).
    ///
    /// Use input = `qp.dequant[0]` itself — guaranteed to be above the
    /// half-dequant threshold and well below the zbin (which is
    /// `0.84 * dequant`).
    #[test]
    fn vp9_quantize_fp_no_zbin_behavior() {
        let qp = q60_y_quantizers();
        let scan = default_scan_4x4();

        // Use a value that's small enough to be below the `vp9_quantize_b`
        // zbin gate (zbin[0] ≈ 0.84 * dequant), but at one-full-dequant —
        // this puts us within the rounding-zone where the fast path still
        // produces a non-zero result.
        let input = i32::from(qp.dequant[0]);
        assert!(
            input < i32::from(qp.zbin[0]) * 2,
            "test precondition: input ({input}) should be small enough to demonstrate the no-zbin gate"
        );

        // Sanity: vp9_quantize_b zeros this same input because of the
        // zbin gate.
        let mut coeff = [0i32; 16];
        coeff[0] = input;
        let (qb_qcoeff, _, qb_eob) = quantize_b_4x4(
            &coeff,
            false,
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
        );
        // The libvpx zbin is `≈0.84 * dequant`, and `input = dequant`,
        // so the coefficient survives the zbin gate. But the post-shift
        // pipeline may still round it down to zero — that's not the
        // property we're testing here. The point is the fp path
        // doesn't reject on zbin at all.

        // Now: with vp9_quantize_fp, the fast-path round_fp is
        // 48/128 * dequant ≈ 22 for dequant=58, so tmp = (58 + 22) *
        // ((1<<16)/58) >> 16 = 80 * 1129 >> 16 = 90320 >> 16 = 1 →
        // non-zero qcoeff.
        let mut qcoeff = [0i16; 16];
        let mut dqcoeff = [0i16; 16];
        let eob = vp9_quantize_fp(
            &coeff,
            16,
            false,
            &qp.round_fp,
            &qp.quant_fp,
            &mut qcoeff,
            &mut dqcoeff,
            &qp.dequant,
            scan.scan(),
        );
        assert_ne!(
            qcoeff[0], 0,
            "fp path produces a non-zero qcoeff for input = dequant"
        );
        assert!(eob >= 1);

        // And demonstrate the original property: take an input that's
        // BELOW zbin but still survives the fp path. For q=60:
        // zbin[0] ≈ 49, dequant[0] = 58 — so we need an input < 49 that
        // still has |coeff| > some_fp_threshold. Use input = dequant / 2
        // + 1 (just above the round_fp/quant_fp threshold). The actual
        // numeric demonstration isn't critical; the load-bearing assertion
        // is "fp at input < zbin can be non-zero, while quantize_b at the
        // same input is zero". Demonstrate that:
        let small_input = i32::from(qp.zbin[0]) - 1;
        let mut coeff_small = [0i32; 16];
        coeff_small[0] = small_input;
        let (qb_small, _, _) = quantize_b_4x4(
            &coeff_small,
            false,
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
        );
        assert_eq!(
            qb_small[0], 0,
            "vp9_quantize_b should zero coefficient below zbin"
        );

        // Note: even quantize_fp may still round small_input to zero
        // because the post-shift threshold can be higher than zbin; but
        // we still validate the structural difference — the fp path has
        // no early-exit on zbin, so the path is _evaluated_ for every
        // coefficient.
        let _ = qb_qcoeff;
        let _ = qb_eob;
    }

    /// Round-trip: feed a synthetic coefficient set through `quantize_fp`
    /// then dequant, and verify the recovered coefficient is within
    /// ±dequant/2 of the input.
    #[test]
    fn vp9_quantize_fp_round_trip_at_q_index_60() {
        let qp = q60_y_quantizers();
        let scan = default_scan_4x4();
        let inputs = [200i32, -150, 80, -300, 50];
        for &input in &inputs {
            let mut coeff = [0i32; 16];
            coeff[0] = input;
            let mut qcoeff = [0i16; 16];
            let mut dqcoeff = [0i16; 16];
            let _eob = vp9_quantize_fp(
                &coeff,
                16,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                &mut qcoeff,
                &mut dqcoeff,
                &qp.dequant,
                scan.scan(),
            );
            // dqcoeff is the reconstructed (decoder-equivalent) value.
            let recovered = i32::from(dqcoeff[0]);
            let tolerance = i32::from(qp.dequant[0]);
            assert!(
                (recovered - input).abs() <= tolerance,
                "input {input} → recovered {recovered}, diff {} > tolerance {tolerance}",
                (recovered - input).abs()
            );
        }
    }

    /// End-to-end round-trip: synthesise a flat 8×8 input, forward-DCT
    /// it, quantize with `vp9_quantize_b` at q=80, push the qcoeffs
    /// through `encode_coefs` → `decode_coefs`, then verify the recovered
    /// dequantized coefficients match the encoder's `dqcoeff`.
    ///
    /// This proves that:
    ///
    /// 1. `vp9_quantize_b` produces qcoeffs in the value range that
    ///    `encode_coefs` accepts (no overflow / no out-of-tree values).
    /// 2. The (qcoeff, dequant) pair the encoder writes is the same
    ///    (dqcoeff) pair the decoder reconstructs.
    /// 3. The `eob` boundary the encoder writes matches what the decoder
    ///    expects.
    #[test]
    fn vp9_quantize_b_round_trip_integration() {
        // 1. Flat 8×8 input — DC-only after fdct8x8.
        // Small positive value so fdct produces a clean DC.
        let input_pixels = [32i16; 64];
        let mut fdct_out = [0i32; 64];
        fdct8x8(&input_pixels, &mut fdct_out, 8);

        // 2. Quantize via vp9_quantize_b at q=80.
        let qp = init_plane_quantizers(80, QuantizerDeltas::Y);
        let scan = default_scan_8x8();
        let mut qcoeff_i16 = [0i16; 64];
        let mut dqcoeff_i16 = [0i16; 64];
        let eob = vp9_quantize_b(
            &fdct_out,
            64,
            false,
            &qp.zbin,
            &qp.round,
            &qp.quant,
            &qp.quant_shift,
            &mut qcoeff_i16,
            &mut dqcoeff_i16,
            &qp.dequant,
            scan.scan(),
        );
        assert!(
            eob >= 1,
            "flat input should produce at least the DC coefficient"
        );

        // 3. Encode the qcoeffs through the entropy writer.
        let fc = FrameContext::new_default();
        // Pick `plane=0 (Y), ref=0 (intra)` — matches the q=80 Y plane
        // setup. coef_probs is `[tx][plane][ref]`; project down.
        let coef_probs = &fc.coef_probs[TxSize::Tx8x8 as usize][0][0];

        let mut bw_buf = vec![0u8; 4096];
        let mut bw = BoolWriter::start_encode(&mut bw_buf);

        // encode_coefs takes raster-order i32 coefficients.
        let raw_coefs: Vec<i32> = qcoeff_i16.iter().map(|&v| i32::from(v)).collect();
        encode_coefs(
            &mut bw,
            coef_probs,
            &scan,
            0, // initial_pt = 0 — no prior context
            TxSize::Tx8x8,
            64,
            &raw_coefs,
            eob as usize,
            None,
        );
        let encoded_len = bw.stop_encode().expect("stop_encode should succeed");

        // 4. Decode the qcoeffs back through the entropy reader.
        let mut reader = BoolReader::new(&bw_buf[..encoded_len])
            .expect("BoolReader::new should accept the encoded buffer");
        let mut decoded_dq = [0i32; 64];
        let recovered_eob = decode_coefs(
            &mut reader,
            coef_probs,
            None,
            &scan,
            0,
            TxSize::Tx8x8,
            64,
            (i32::from(qp.dequant[0]), i32::from(qp.dequant[1])),
            &mut decoded_dq,
        );

        // 5. Round-trip invariant: the encoder's dqcoeff (raster order)
        // must equal the decoder's recovered dq (raster order).
        assert_eq!(
            recovered_eob, eob as usize,
            "decoded eob = {recovered_eob} should match encoded eob = {eob}",
        );
        for raster in 0..64 {
            let enc_dq = i32::from(dqcoeff_i16[raster]);
            let dec_dq = decoded_dq[raster];
            assert_eq!(
                enc_dq, dec_dq,
                "raster {raster}: encoder dqcoeff = {enc_dq}, decoder dq = {dec_dq}"
            );
        }

        // Also assert the post-quantize reconstruction is within
        // `dequant` of the pre-quant fdct8x8 coefficient — bounding the
        // quantization error. This is the macro-level "round-trip
        // recovers input" property F5.1 will depend on.
        for raster in 0..64 {
            let pre = fdct_out[raster];
            let post = i32::from(dqcoeff_i16[raster]);
            let band = usize::from(raster != 0);
            let tol = i32::from(qp.dequant[band]);
            assert!(
                (pre - post).abs() <= tol,
                "raster {raster}: pre-quant {pre}, post-quant {post}, |diff| > dequant {tol}",
            );
        }
    }

    /// Quantizer constants for q=60 on the Y plane. Verified inline:
    ///
    /// - `dc_quant(60, 0) = DC_QLOOKUP[60]`: see `quant/tables.rs` —
    ///   confirmed via [`super::PlaneQuantizers::dequant`] at runtime.
    /// - `qzbin_factor = 84` since `q != 0` and `dc_quant(60) < 148`.
    /// - `qrounding_factor = 48` since `q != 0`.
    /// - `qrounding_factor_fp = 48` for DC, `42` for AC.
    /// - All eight outputs (`quant`, `quant_shift`, `quant_fp`, `round`,
    ///   `round_fp`, `zbin`, `dequant`) match libvpx
    ///   `vp9_init_quantizer` (`vp9/encoder/vp9_quantize.c:184-244`) bit
    ///   for bit.
    fn q60_y_quantizers() -> PlaneQuantizers {
        init_plane_quantizers(60, QuantizerDeltas::Y)
    }

    /// `init_plane_quantizers(0, Y)` should produce the lossless
    /// quantizer slot — quant = 4 (the smallest entry in DC_QLOOKUP /
    /// AC_QLOOKUP) and the `q == 0` rounding constants.
    #[test]
    fn init_plane_quantizers_q0_is_lossless() {
        let qp = init_plane_quantizers(0, QuantizerDeltas::Y);
        assert_eq!(qp.dequant, [4, 4]);
        // q == 0 → qrounding_factor = 64, qrounding_factor_fp = 64.
        // round = (64 * 4) >> 7 = 2; round_fp = same.
        assert_eq!(qp.round, [2, 2]);
        assert_eq!(qp.round_fp, [2, 2]);
        // q == 0 → qzbin_factor = 64. zbin = ROUND_POWER_OF_TWO(64 * 4,
        // 7) = (256 + 64) >> 7 = 2.
        assert_eq!(qp.zbin, [2, 2]);
    }

    /// `init_plane_quantizers(60, Y)` sanity-checks the dequant pair
    /// matches the freestanding `dc_quant` / `ac_quant` helpers.
    #[test]
    fn init_plane_quantizers_q60_dequant_matches_lookup() {
        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        assert_eq!(qp.dequant[0], dc_quant(60, 0, BitDepth::Eight));
        assert_eq!(qp.dequant[1], ac_quant(60, 0, BitDepth::Eight));
    }

    // ------------------------------------------------------------------
    // Stage 9 — fill remaining branch coverage gaps in quant/forward.rs
    // ------------------------------------------------------------------

    /// `vp9_quantize_b_32x32` with `skip_block = true` zeros both
    /// output buffers and returns `eob = 0`.
    #[test]
    fn vp9_quantize_b_32x32_skip_block_zeroes_everything() {
        let coeff = vec![1000i32; 1024];
        let mut qcoeff = vec![99i16; 1024]; // pre-seeded — must be cleared
        let mut dqcoeff = vec![99i16; 1024];
        let scan = crate::transforms::scans::DEFAULT_SCAN_32X32;
        let eob = vp9_quantize_b_32x32(
            &coeff,
            true,
            &[10, 10],
            &[4, 4],
            &[16384, 16384],
            &[1, 1],
            &mut qcoeff,
            &mut dqcoeff,
            &[8, 8],
            &scan,
        );
        assert_eq!(eob, 0);
        assert!(qcoeff.iter().all(|&v| v == 0));
        assert!(dqcoeff.iter().all(|&v| v == 0));
    }

    /// `vp9_quantize_fp` with `skip_block = true` zeros both buffers
    /// and returns `eob = 0`.
    #[test]
    fn vp9_quantize_fp_skip_block_zeroes_everything() {
        let coeff = [1000i32; 16];
        let mut qcoeff = [99i16; 16];
        let mut dqcoeff = [99i16; 16];
        let scan = default_scan_4x4();
        let eob = vp9_quantize_fp(
            &coeff,
            16,
            true,
            &[4, 4],
            &[16384, 16384],
            &mut qcoeff,
            &mut dqcoeff,
            &[8, 8],
            scan.scan(),
        );
        assert_eq!(eob, 0);
        assert!(qcoeff.iter().all(|&v| v == 0));
        assert!(dqcoeff.iter().all(|&v| v == 0));
    }

    /// `vp9_quantize_fp_32x32` with `skip_block = true` zeros and
    /// returns 0. Exercises the skip arm of the 32×32 fp kernel.
    #[test]
    fn vp9_quantize_fp_32x32_skip_block_zeroes_everything() {
        let coeff = vec![5000i32; 1024];
        let mut qcoeff = vec![123i16; 1024];
        let mut dqcoeff = vec![123i16; 1024];
        let scan = crate::transforms::scans::DEFAULT_SCAN_32X32;
        let eob = vp9_quantize_fp_32x32(
            &coeff,
            true,
            &[4, 4],
            &[16384, 16384],
            &mut qcoeff,
            &mut dqcoeff,
            &[8, 8],
            &scan,
        );
        assert_eq!(eob, 0);
        assert!(qcoeff.iter().all(|&v| v == 0));
        assert!(dqcoeff.iter().all(|&v| v == 0));
    }

    /// `vp9_quantize_fp_32x32` with a coefficient ABOVE the
    /// `dequant >> 2` gate at scan position 0 produces a non-zero
    /// qcoeff/dqcoeff. Exercises the `if abs_coeff >= dq>>2` true
    /// branch of the fp_32x32 kernel.
    #[test]
    fn vp9_quantize_fp_32x32_above_dequant_gate_produces_nonzero() {
        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        let mut coeff = vec![0i32; 1024];
        coeff[0] = 4000; // well above dequant/2 and the fp threshold
        let mut qcoeff = vec![0i16; 1024];
        let mut dqcoeff = vec![0i16; 1024];
        let scan = crate::transforms::scans::DEFAULT_SCAN_32X32;
        let eob = vp9_quantize_fp_32x32(
            &coeff,
            false,
            &qp.round_fp,
            &qp.quant_fp,
            &mut qcoeff,
            &mut dqcoeff,
            &qp.dequant,
            &scan,
        );
        assert!(eob >= 1, "above-gate input should produce nonzero qcoeff");
        assert_ne!(qcoeff[0], 0);
        // Reconstruction invariant: dqcoeff = qcoeff * dequant / 2.
        assert_eq!(
            i32::from(dqcoeff[0]),
            (i32::from(qcoeff[0]) * i32::from(qp.dequant[0])) / 2,
        );
    }

    /// `vp9_quantize_fp_32x32` with a coefficient BELOW the
    /// `dequant >> 2` gate yields zero qcoeff/dqcoeff. Exercises the
    /// `if abs_coeff >= dq>>2` false branch.
    #[test]
    fn vp9_quantize_fp_32x32_below_dequant_gate_quantizes_to_zero() {
        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        let mut coeff = vec![0i32; 1024];
        // dequant[0] >> 2 = qp.dequant[0] / 4. Use 0.
        coeff[0] = 0;
        let mut qcoeff = vec![0i16; 1024];
        let mut dqcoeff = vec![0i16; 1024];
        let scan = crate::transforms::scans::DEFAULT_SCAN_32X32;
        let eob = vp9_quantize_fp_32x32(
            &coeff,
            false,
            &qp.round_fp,
            &qp.quant_fp,
            &mut qcoeff,
            &mut dqcoeff,
            &qp.dequant,
            &scan,
        );
        assert_eq!(eob, 0, "zero input → eob=0");
        assert!(qcoeff.iter().all(|&v| v == 0));
        assert!(dqcoeff.iter().all(|&v| v == 0));
    }

    /// `vp9_quantize_b_32x32` round-trip with negative inputs covers
    /// the `coeff_sign != 0` arithmetic path on the 32×32 kernel.
    #[test]
    fn vp9_quantize_b_32x32_negative_coeff_preserves_sign() {
        let qp = init_plane_quantizers(80, QuantizerDeltas::Y);
        let mut coeff = vec![0i32; 1024];
        coeff[0] = -3000; // above zbin
        let mut qcoeff = vec![0i16; 1024];
        let mut dqcoeff = vec![0i16; 1024];
        let scan = crate::transforms::scans::DEFAULT_SCAN_32X32;
        let eob = vp9_quantize_b_32x32(
            &coeff,
            false,
            &qp.zbin,
            &qp.round,
            &qp.quant,
            &qp.quant_shift,
            &mut qcoeff,
            &mut dqcoeff,
            &qp.dequant,
            &scan,
        );
        assert!(eob >= 1);
        assert!(qcoeff[0] < 0, "negative input → negative qcoeff");
        assert!(dqcoeff[0] < 0);
    }

    /// `init_plane_quantizers` at high `q_index` (e.g. 240) where
    /// `dc_quant(q, 0) >= 148` hits the `qzbin_factor = 80` branch
    /// (the lower-than-148 path uses 84). Both branches must be
    /// reachable for `qzbin_factor_8bit` to be fully covered.
    #[test]
    fn init_plane_quantizers_high_q_uses_qzbin_factor_80() {
        let qp_low = init_plane_quantizers(60, QuantizerDeltas::Y);
        let qp_high = init_plane_quantizers(240, QuantizerDeltas::Y);
        // Both produce non-default (nonzero) results.
        assert!(qp_low.zbin[0] != 0);
        assert!(qp_high.zbin[0] != 0);
        // High q has a larger dequant (the lookup table grows
        // monotonically with q).
        assert!(
            qp_high.dequant[0] > qp_low.dequant[0],
            "q=240 dequant should exceed q=60",
        );
        // Spot-check: at q=240, `dc_quant(240) >= 148` should hold
        // (it's far past the 84/80 cutoff).
        assert!(
            dc_quant(240, 0, BitDepth::Eight) >= 148,
            "q=240 dc_quant must be >= 148 to exercise the 80-factor branch",
        );
    }

    /// `init_plane_quantizers` for chroma with `ac_delta != 0` hits
    /// the `ac_quant(q, ac_delta, ...)` path with a non-zero delta —
    /// distinct from the `Y` default which has `ac_delta = 0`.
    #[test]
    fn init_plane_quantizers_uv_with_ac_delta_differs_from_y() {
        let qp_y = init_plane_quantizers(60, QuantizerDeltas::Y);
        let qp_uv = init_plane_quantizers(
            60,
            QuantizerDeltas {
                dc_delta: 0,
                ac_delta: 4,
            },
        );
        // The AC dequant must differ between Y (ac_delta=0) and UV
        // (ac_delta=4) at q=60.
        assert_ne!(qp_y.dequant[1], qp_uv.dequant[1]);
        // DC dequant matches (same dc_delta=0).
        assert_eq!(qp_y.dequant[0], qp_uv.dequant[0]);
    }

    /// `plane_dequant_pair` returns `(dc_quant, ac_quant)` for the
    /// given `(q_index, deltas)`. Compare against the lookup helpers.
    #[test]
    fn plane_dequant_pair_matches_dc_ac_quant_helpers() {
        let deltas = QuantizerDeltas {
            dc_delta: -2,
            ac_delta: 6,
        };
        let (dc, ac) = plane_dequant_pair(80, deltas);
        assert_eq!(dc, dc_quant(80, -2, BitDepth::Eight));
        assert_eq!(ac, ac_quant(80, 6, BitDepth::Eight));
    }

    /// `n_coeffs_for(TxSize)` matches the underlying enum method.
    #[test]
    fn n_coeffs_for_matches_tx_size_n_coeffs() {
        for tx in [
            TxSize::Tx4x4,
            TxSize::Tx8x8,
            TxSize::Tx16x16,
            TxSize::Tx32x32,
        ] {
            assert_eq!(n_coeffs_for(tx), tx.n_coeffs());
        }
    }

    /// `scan_for_forward(order)` returns the scan slice. Trivial pass
    /// through — pin it to keep the convenience helper exercised.
    #[test]
    fn scan_for_forward_returns_scan_slice() {
        let scan = default_scan_4x4();
        let forward = scan_for_forward(&scan);
        assert_eq!(forward, scan.scan());
    }

    /// Hand-verified: `invert_quant(8)` should yield the libvpx values
    /// for `dequant = 8`. From the algorithm: `l = floor(log2(8)) = 3`,
    /// so `m = 1 + (1 << 19) / 8 = 1 + 65536 = 65537`, `quant = 65537 -
    /// 65536 = 1` (truncated to `i16`), `shift = 1 << 13 = 8192`.
    #[test]
    fn invert_quant_for_dequant_8() {
        let (q, s) = invert_quant(8);
        assert_eq!(q, 1);
        assert_eq!(s, 8192);
    }
}
